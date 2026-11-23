"""
Preprocessing dataset pembatalan pesanan penginapan.

Semua statistik (median, daftar kategori, mean, dan standar deviasi) dihitung di fit dari data
latih saja, lalu transform memakai statistik yang sama untuk data validasi dan test. Dengan begitu
tidak ada informasi dari data validasi atau test yang bocor ke model.

Urutan langkah mengikuti slide Data Preparation IF3170 (Materi 09):
1. Imputasi missing value: median untuk numerik, kategori "Unknown" untuk kategorikal.
2. Penambahan fitur turunan.
3. One-hot encoding; kategori yang jarang digabung menjadi "Other".
4. Standarisasi z-score untuk kolom numerik.

Catatan cleaning: drop_duplicate_rows dan drop_invalid_rows hanya boleh dipakai pada data latih.
Data test tidak boleh dibuang barisnya karena submission harus memuat semua id.
"""

import pickle

import numpy as np
import pandas as pd

#: Nama kolom target.
TARGET = "is_canceled"

#: Nama kolom identitas baris, tidak dipakai sebagai fitur.
ID_COL = "id"

#: Fitur numerik asli dataset.
NUMERIC_COLS = [
    "num_adults", "num_children", "weekend_nights", "weekday_nights",
    "needs_parking", "days_before_arrival", "arrival_year", "arrival_month",
    "arrival_day", "is_returning_guest", "prior_cancellations",
    "prior_completed_bookings", "nightly_rate_idr", "special_requests",
]

#: Fitur kategorikal nominal yang di-one-hot.
CATEGORICAL_COLS = ["meal_package", "room_category", "booking_channel"]

#: Fitur turunan yang ditambahkan oleh Preprocessor._derive.
DERIVED_COLS = ["log_lead_time", "arrival_date_ordinal"]

def drop_duplicate_rows(df):
    """
    Membuang baris yang semua kolomnya, selain id, sama dengan baris sebelumnya.
    Hanya untuk data latih.

    :param df: DataFrame data
    :return: DataFrame tanpa duplikat dengan indeks baru 0..n-1
    """
    cols = [c for c in df.columns if c != ID_COL]
    return df.loc[~df.duplicated(subset=cols)].reset_index(drop=True)

def invalid_row_report(df):
    """
    Menghitung baris yang mencurigakan atau tidak valid tanpa mengubah data. Cocok dipakai di
    notebook untuk mendokumentasikan tahap cleaning pada data latih maupun test.

    Nilai yang hilang tidak dihitung karena ditangani oleh imputasi.

    :param df: DataFrame data
    :return: dict jumlah baris per kategori:
             zero_nights (weekend_nights + weekday_nights = 0, tidak valid untuk sebuah pesanan),
             zero_adults (num_adults = 0),
             zero_rate (nightly_rate_idr = 0, mungkin tamu complimentary sehingga bisa valid),
             returning_without_history (tamu returning tanpa pesanan selesai sebelumnya)
    """
    nights = df["weekend_nights"] + df["weekday_nights"]
    returning = (df["is_returning_guest"] == 1) & (df["prior_completed_bookings"] == 0)
    return {
        "zero_nights": int((nights == 0).sum()),
        "zero_adults": int((df["num_adults"] == 0).sum()),
        "zero_rate": int((df["nightly_rate_idr"] == 0).sum()),
        "returning_without_history": int(returning.sum()),
    }

def drop_invalid_rows(df):
    """
    Membuang pesanan dengan 0 malam menginap, yaitu weekend_nights + weekday_nights = 0.
    Kategori lain di invalid_row_report sengaja tidak dibuang karena masih bisa valid
    (misalnya tamu complimentary bertarif 0). Hanya untuk data latih.

    Fungsi ini tidak dipanggil oleh Preprocessor. Jika dipakai pada data latih final, hasil model
    berubah, sehingga submission Kaggle harus dihasilkan ulang dan diunggah ulang.

    :param df: DataFrame data latih
    :return: DataFrame tanpa baris 0 malam dengan indeks baru 0..n-1
    """
    nights = df["weekend_nights"] + df["weekday_nights"]
    return df.loc[~(nights == 0)].reset_index(drop=True)

class Preprocessor:
    """
    Pipeline preprocessing yang di-fit pada data latih dan dapat mentransformasi data mana pun
    menjadi matriks NumPy. Kolom numerik berada di depan, diikuti blok one-hot setiap kolom
    kategorikal sesuai urutan CATEGORICAL_COLS.
    """

    def __init__(self, scale=True, min_category_freq=20, add_derived=True, feature_weights=None):
        """
        :param scale: standarisasi kolom numerik; dibutuhkan model berbasis jarak dan gradien
                      seperti KNN, Logistic Regression, dan SVM
        :param min_category_freq: kategori yang muncul kurang dari nilai ini di data latih digabung
                                  menjadi "Other", misalnya Full Board yang hanya 4 baris
        :param add_derived: tambahkan kolom DERIVED_COLS
        :param feature_weights: pengali setelah scaling dengan key nama kolom numerik atau nama
                                kolom kategorikal (berlaku untuk semua kolom one-hot-nya). Bobot
                                lebih besar membuat fitur itu lebih berpengaruh pada jarak KNN
                                (feature weighting, Cunningham dan Delany, arXiv:2004.04523, Section 3.1)
        """
        self.scale = scale
        self.min_category_freq = min_category_freq
        self.add_derived = add_derived
        self.feature_weights = dict(feature_weights or {})
        self.fitted_ = False

    @staticmethod
    def _derive(df):
        """
        Membuat fitur turunan. Harus dipanggil setelah kolom numerik diimputasi.

        :param df: DataFrame kolom numerik tanpa missing value
        :return: DataFrame berisi kolom DERIVED_COLS
        """
        out = pd.DataFrame(index=df.index)
        # Lead time sangat miring ke kanan (0 sampai 448 hari), log memampatkan ekornya.
        out["log_lead_time"] = np.log1p(df["days_before_arrival"])
        # Tanggal kedatangan dijadikan satu angka agar tren waktu (2023 vs 2024) terwakili satu dimensi.
        dates = pd.to_datetime(
            dict(year=df["arrival_year"], month=df["arrival_month"], day=df["arrival_day"]),
            errors="coerce",
        )
        out["arrival_date_ordinal"] = dates.map(
            lambda d: float(d.toordinal()) if pd.notna(d) else np.nan
        )
        return out

    def _impute_numeric(self, df):
        """
        :param df: DataFrame data
        :return: kolom NUMERIC_COLS dengan missing value diganti median data latih
        """
        return df[NUMERIC_COLS].astype(float).fillna(self.medians_)

    def _clean_categorical(self, df):
        """
        :param df: DataFrame data
        :return: kolom CATEGORICAL_COLS dengan missing value menjadi "Unknown" dan kategori
                 yang tidak dikenal saat fit menjadi "Other"
        """
        cat = df[CATEGORICAL_COLS].fillna("Unknown").astype(str)
        for c in CATEGORICAL_COLS:
            known = set(self.categories_[c])
            cat[c] = cat[c].where(cat[c].isin(known), "Other")
        return cat

    def fit(self, df):
        """
        Mempelajari median, daftar kategori, mean, dan standar deviasi dari data latih.

        :param df: DataFrame data latih
        :return: objek ini sendiri
        """
        self.medians_ = df[NUMERIC_COLS].astype(float).median()

        # Hanya kategori yang cukup sering muncul yang mendapat kolom one-hot sendiri.
        self.categories_ = {}
        cat = df[CATEGORICAL_COLS].fillna("Unknown").astype(str)
        for c in CATEGORICAL_COLS:
            counts = cat[c].value_counts()
            keep = sorted(counts[counts >= self.min_category_freq].index)
            self.categories_[c] = keep + ["Other"]

        num = self._impute_numeric(df)
        if self.add_derived:
            derived = self._derive(num)
            self.derived_medians_ = derived.median()
            num = pd.concat([num, derived.fillna(self.derived_medians_)], axis=1)

        self.numeric_features_ = list(num.columns)
        self.mean_ = num.mean()
        # Kolom konstan memiliki std 0; diganti 1 supaya tidak terjadi pembagian dengan nol.
        self.std_ = num.std(ddof=0).replace(0, 1.0)

        self.feature_names_ = list(self.numeric_features_)
        for c in CATEGORICAL_COLS:
            for v in self.categories_[c]:
                self.feature_names_.append(f"{c}={v}")

        self.fitted_ = True
        return self

    def transform(self, df):
        """
        :param df: DataFrame data latih, validasi, atau test
        :return: matriks fitur berukuran (n, len(feature_names_))
        :raises RuntimeError: jika fit belum dipanggil
        """
        if not self.fitted_:
            raise RuntimeError("Preprocessor belum di-fit, panggil fit sebelum transform")

        num = self._impute_numeric(df)
        if self.add_derived:
            derived = self._derive(num).fillna(self.derived_medians_)
            num = pd.concat([num, derived], axis=1)
        num = num[self.numeric_features_]
        if self.scale:
            num = (num - self.mean_) / self.std_
        for col, w in self.feature_weights.items():
            if col in num.columns:
                num[col] = num[col] * w

        cat = self._clean_categorical(df)
        blocks = [num.to_numpy(dtype=float)]
        for c in CATEGORICAL_COLS:
            values = cat[c].to_numpy(dtype=object)[:, None]
            cats = np.array(self.categories_[c], dtype=object)[None, :]
            onehot = (values == cats).astype(float)
            blocks.append(onehot * self.feature_weights.get(c, 1.0))
        return np.hstack(blocks)

    def fit_transform(self, df):
        """
        :param df: DataFrame data latih
        :return: matriks fitur data latih
        """
        return self.fit(df).transform(df)

    def save(self, path):
        """
        :param path: lokasi berkas pickle tujuan
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        """
        :param path: lokasi berkas pickle hasil save
        :return: objek Preprocessor yang tersimpan
        """
        with open(path, "rb") as f:
            return pickle.load(f)

class NaiveBayesPreprocessor:
    """
    Preprocessing khusus Naive Bayes. Slide [5] (Maulidevi, Bayesian Network as Knowledge
    Representation, IF3170 2024, slide 27-42) hanya membahas atribut kategorikal, sehingga
    atribut numerik didiskretkan menjadi bin kuantil. Semua statistik dipelajari dari data latih.

    Hasil transform adalah matriks kode kategori bertipe int64:
    - fitur numerik: indeks bin 0..n_bin-1 (missing diimputasi median data latih lebih dulu),
    - fitur kategorikal: 0 untuk missing, 1 untuk kategori yang tidak ada di data latih,
      dan 2+ untuk kategori data latih.
    """

    def __init__(self, n_bins=10):
        """
        :param n_bins: jumlah bin kuantil maksimum per fitur numerik, minimal 2
        :raises ValueError: jika n_bins bukan integer minimal 2
        """
        if not isinstance(n_bins, (int, np.integer)) or isinstance(n_bins, bool) or n_bins < 2:
            raise ValueError("n_bins harus integer minimal 2")
        self.n_bins = n_bins
        self.fitted_ = False

    def _numeric(self, df):
        numeric = df[NUMERIC_COLS].astype(float)
        if np.isinf(numeric.to_numpy()).any():
            raise ValueError("Fitur numerik tidak boleh mengandung infinity")
        return numeric

    def fit(self, df):
        """
        Mempelajari median, batas bin kuantil, dan daftar kategori dari data latih.

        :param df: DataFrame data latih
        :return: objek ini sendiri
        :raises ValueError: jika data kosong atau ada nilai infinity
        """
        self.fitted_ = False
        if len(df) == 0:
            raise ValueError("Data latih tidak boleh kosong")
        numeric = self._numeric(df)
        # Kolom yang seluruhnya missing memakai 0 tanpa melihat validasi/test.
        self.medians_ = numeric.median().fillna(0.0)
        numeric = numeric.fillna(self.medians_)
        self.bin_edges_ = {}
        sizes = []
        for column in NUMERIC_COLS:
            values = numeric[column].to_numpy()
            unique = np.unique(values)
            if len(unique) <= self.n_bins:
                edges = unique[:-1] / 2 + unique[1:] / 2
            else:
                edges = np.unique(np.quantile(values, np.linspace(0, 1, self.n_bins + 1)[1:-1]))
                edges = edges[(edges > unique[0]) & (edges < unique[-1])]
            self.bin_edges_[column] = edges
            sizes.append(len(edges) + 1)
        # Kode 0 untuk missing, 1 untuk unseen, dan 2+ untuk kategori training.
        self.category_maps_ = {}
        for column in CATEGORICAL_COLS:
            categories = sorted(df[column].dropna().astype(str).unique())
            self.category_maps_[column] = {value: i + 2 for i, value in enumerate(categories)}
            sizes.append(len(categories) + 2)
        self.feature_names_ = NUMERIC_COLS + CATEGORICAL_COLS
        self.n_categories_ = np.asarray(sizes, dtype=np.int64)
        self.fitted_ = True
        return self

    def transform(self, df):
        """
        :param df: DataFrame data latih, validasi, atau test
        :return: matriks kode kategori int64 berukuran (n, len(feature_names_))
        :raises RuntimeError: jika fit belum dipanggil
        """
        if not self.fitted_:
            raise RuntimeError("NaiveBayesPreprocessor belum di-fit")
        numeric = self._numeric(df).fillna(self.medians_)
        columns = [np.searchsorted(self.bin_edges_[column], numeric[column], side="right")
                   for column in NUMERIC_COLS]
        for column in CATEGORICAL_COLS:
            values = df[column]
            codes = values.astype(str).map(self.category_maps_[column]).fillna(1)
            codes = codes.where(values.notna(), 0)
            columns.append(codes.to_numpy(dtype=np.int64))
        return np.column_stack(columns).astype(np.int64)

    def fit_transform(self, df):
        """
        :param df: DataFrame data latih
        :return: matriks kode kategori data latih
        """
        return self.fit(df).transform(df)

    def save(self, path):
        """
        :param path: lokasi berkas pickle tujuan
        :raises RuntimeError: jika belum di-fit
        """
        if not self.fitted_:
            raise RuntimeError("NaiveBayesPreprocessor belum di-fit")
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        """
        :param path: lokasi berkas pickle hasil save
        :return: objek NaiveBayesPreprocessor yang tersimpan
        """
        with open(path, "rb") as stream:
            return pickle.load(stream)

class SVMPreprocessor(Preprocessor):
    """
    Preprocessing khusus SVM.

    Parameter
    ---------
    log_history : bool
        Transformasi log1p untuk prior_cancellations dan prior_completed_bookings
        agar distribusi ekornya lebih stabil untuk kernel Gaussian dan Polynomial.
    binary_unscaled : bool
        Fitur biner (needs_parking, is_returning_guest) tidak di-z-score sehingga
        tetap bernilai 0/1. Untuk kernel linear dampaknya kecil, tetapi untuk kernel
        Gaussian dan Polynomial hal ini mempengaruhi skala efektif gamma dan C.
        Pilihan ini dicatat di laporan sebagai hyperparameter tetap.
    date_encoding : {"original", "ordinal", "cyclical"}
        Cara menyandikan arrival_year, arrival_month, arrival_day. Mode cyclical
        menggantikan ketiga kolom dengan sin/cos bulan dan hari dalam minggu.
    """

    def __init__(self, log_history=True, binary_unscaled=True, date_encoding="original"):
        if date_encoding not in ("original", "ordinal", "cyclical"):
            raise ValueError("date_encoding harus 'original', 'ordinal', atau 'cyclical'")
        super().__init__(feature_weights=None)
        self.log_history = log_history
        self.binary_unscaled = binary_unscaled
        self.date_encoding = date_encoding

    def _impute_numeric(self, df):
        num = super()._impute_numeric(df)
        if self.log_history:
            columns = ["prior_cancellations", "prior_completed_bookings"]
            if (num[columns] < 0).any().any():
                raise ValueError("Riwayat booking tidak boleh negatif")
            num[columns] = np.log1p(num[columns])
        return num

    def _derive(self, df):
        out = super()._derive(df)
        if self.date_encoding == "cyclical":
            dates = pd.to_datetime(
                dict(year=df["arrival_year"], month=df["arrival_month"],
                     day=df["arrival_day"]), errors="coerce")
            month_angle = 2 * np.pi * (df["arrival_month"] - 1) / 12
            weekday_angle = 2 * np.pi * dates.dt.dayofweek / 7
            out["arrival_month_sin"] = np.sin(month_angle)
            out["arrival_month_cos"] = np.cos(month_angle)
            out["arrival_weekday_sin"] = np.sin(weekday_angle)
            out["arrival_weekday_cos"] = np.cos(weekday_angle)
        return out

    def fit(self, df):
        super().fit(df)
        if self.date_encoding != "original":
            removed = {"arrival_year", "arrival_month", "arrival_day"}
            self.numeric_features_ = [c for c in self.numeric_features_ if c not in removed]
            self.feature_names_ = [c for c in self.feature_names_ if c not in removed]
            self.mean_ = self.mean_[self.numeric_features_]
            self.std_ = self.std_[self.numeric_features_]
        if self.binary_unscaled:
            binary = ["needs_parking", "is_returning_guest"]
            self.mean_[binary] = 0.0
            self.std_[binary] = 1.0
        return self
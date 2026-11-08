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

    :param df: DataFrame data
    :return: DataFrame tanpa duplikat dengan indeks baru 0..n-1
    """
    cols = [c for c in df.columns if c != ID_COL]
    return df.loc[~df.duplicated(subset=cols)].reset_index(drop=True)


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

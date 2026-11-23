from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.ticker import FormatStrFormatter


def log_loss_grid(feature, labels, intercepts, weights):
    """Hitung rata-rata log-loss untuk tiap pasangan (intersep, bobot).
    Hasil berbentuk (jumlah bobot, jumlah intersep).
    """
    feature = np.asarray(feature, dtype=float)
    labels = np.asarray(labels)
    intercepts = np.asarray(intercepts, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if feature.ndim != 1 or not feature.size or not np.isfinite(feature).all():
        raise ValueError("Fitur harus berupa vektor numerik finite yang tidak kosong.")
    if labels.shape != feature.shape or not np.isin(labels, [0, 1]).all():
        raise ValueError("Setiap nilai fitur harus memiliki label biner 0 atau 1.")
    if any(axis.ndim != 1 or not axis.size or not np.isfinite(axis).all() for axis in (intercepts, weights)):
        raise ValueError("Sumbu parameter harus berupa vektor finite yang tidak kosong.")

    signs = np.where(labels == 1, -1.0, 1.0)
    signed_feature = signs * feature
    signed_intercepts = intercepts[:, None] * signs
    loss = np.empty((weights.size, intercepts.size))
    for row, weight in enumerate(weights):
        signed_scores = signed_intercepts + weight * signed_feature
        np.logaddexp(0.0, signed_scores, out=signed_scores)
        loss[row] = signed_scores.mean(axis=1)
    return loss


def _ffmpeg_path():
    if FFMpegWriter.isAvailable():
        return FFMpegWriter.bin_path()

    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        return get_ffmpeg_exe()
    except (ImportError, RuntimeError) as error:
        raise RuntimeError(
            "FFmpeg diperlukan untuk MP4. Tambahkan FFmpeg ke PATH atau "
            "jalankan %pip install imageio-ffmpeg pada notebook ini."
        ) from error


def save_loss_contour(feature, labels, theta_history, loss_history, output_path, *, feature_name, preview_path=None):
    theta_history = np.asarray(theta_history, dtype=float)
    loss_history = np.asarray(loss_history, dtype=float)
    if theta_history.ndim != 2 or theta_history.shape[1] != 2 or len(theta_history) < 2:
        raise ValueError("Riwayat harus berisi dua parameter dan minimal satu epoch training.")
    if loss_history.shape != (len(theta_history),) or not (np.isfinite(theta_history).all() and np.isfinite(loss_history).all()):
        raise ValueError("Parameter dan log-loss harus finite pada setiap epoch yang sama.")

    ffmpeg = _ffmpeg_path()
    padded = np.maximum(0.25 * np.ptp(theta_history, axis=0), 0.16)
    lower = theta_history.min(axis=0) - padded
    upper = theta_history.max(axis=0) + padded
    intercepts, weights = np.linspace(lower, upper, 101).T
    loss = log_loss_grid(feature, labels, intercepts, weights)
    levels = loss.min() + (loss.max() - loss.min()) * np.linspace(0, 1, 11)**1.7

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps = 24
    epochs = np.arange(len(theta_history))
    # Ulangi frame agar setiap titik sempat terlihat tanpa menambah titik training.
    frames = np.pad(np.repeat(epochs, 6), (18, 30), mode='edge')
    accent = '#b43d16'
    ink = '#14293a'
    fig, (contour_ax, history_ax) = plt.subplots(
        1, 2, figsize=(12.8, 7.2), width_ratios=(1.85, 1),
        facecolor='#f5f8fb', gridspec_kw={'wspace': 0.42},
    )
    try:
        fig.subplots_adjust(left=0.085, right=0.955, top=0.78, bottom=0.14)
        fig.text(0.085, 0.94, 'Lintasan training Logistic Regression', fontsize=24, weight='bold', color=ink)
        fig.text(0.085, 0.89, f'{feature_name} terstandardisasi | satu titik per epoch', fontsize=12, color=ink)

        contour = contour_ax.contourf(intercepts, weights, loss, levels=levels, cmap='Blues')
        contour_ax.contour(intercepts, weights, loss, levels=levels[1:-1], colors='#356789', linewidths=0.65)
        scale = fig.colorbar(contour, ax=contour_ax, pad=0.025, ticks=levels[::2], format='%.3f')
        scale.set_label('Log-loss latih')
        contour_ax.set(xlabel=r'Intersep $\theta_0$', ylabel=r'Bobot fitur $\theta_1$', title='Kontur parameter')
        path, = contour_ax.plot([], [], '-o', color=accent, lw=2, ms=3.5, zorder=4, label='Lintasan')
        contour_ax.scatter(*theta_history[0], s=95, facecolor='white', edgecolor=ink, linewidth=1.8, marker='s', label='Awal', zorder=5)
        position, = contour_ax.plot([], [], 'o', color=accent, ms=9, markeredgecolor='white', markeredgewidth=1.5, zorder=6, label='Epoch saat ini')
        contour_ax.legend(loc='upper right', framealpha=0.95)

        history_ax.set(xlim=(-0.8, epochs[-1] + 0.8), title='Perubahan log-loss', xlabel='Epoch', ylabel='Log-loss latih')
        loss_padding = max(np.ptp(loss_history) * 0.12, 0.001)
        history_ax.set_ylim(loss_history.min() - loss_padding, loss_history.max() + loss_padding)
        history_ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        history_ax.grid(color='#dbe5ee', linewidth=0.8)
        history_ax.set_axisbelow(True)
        history_line, = history_ax.plot([], [], color=accent, lw=2)
        history_point, = history_ax.plot([], [], 'o', color=accent, ms=8, markeredgecolor='white')
        status = fig.text(0.085, 0.06, '', fontsize=12, color=ink)

        def update(epoch):
            path.set_data(*theta_history[:epoch + 1].T)
            position.set_data(*theta_history[epoch:epoch + 1].T)
            history_line.set_data(epochs[:epoch + 1], loss_history[:epoch + 1])
            history_point.set_data([epoch], [loss_history[epoch]])
            status.set_text(f'Epoch {epoch:02d}/{epochs[-1]:02d}     |     Log-loss latih {loss_history[epoch]:.5f}')

        animation = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, repeat=False)
        with mpl.rc_context({'animation.ffmpeg_path': ffmpeg}), TemporaryDirectory(dir=output_path.parent) as tmp:
            video_path = Path(tmp) / 'animation.mp4'
            writer = FFMpegWriter(fps=fps, codec='libx264', extra_args=['-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
            animation.save(video_path, writer=writer, dpi=100)
            video_path.replace(output_path)

        if preview_path is not None:
            update(epochs[-1])
            fig.savefig(preview_path, dpi=100)
    finally:
        plt.close(fig)
    return output_path

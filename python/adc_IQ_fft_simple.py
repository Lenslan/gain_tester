"""
ADC IQ FFT 分析工具 V1.2 - pyFFTW 加速版

数据排布（每4行循环）：
  [0] Path0_I
  [1] Path0_Q
  [2] Path1_I
  [3] Path1_Q

选项：
- 路径选择: Path0 或 Path1
- 数据选择: I only, Q only, I&Q
"""

import sys
import os
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLineEdit, QTextEdit,
                             QLabel, QComboBox, QCheckBox, QFileDialog, QMessageBox, QGroupBox)
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtGui import QTextCursor, QIcon

try:
    import matplotlib.pyplot as plt
    import matplotlib
    # 配置中文字体
    import platform
    if platform.system() == 'Windows':
        matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
    matplotlib.rcParams['axes.unicode_minus'] = False
    MATPLOTLIB_AVAILABLE = True
except:
    MATPLOTLIB_AVAILABLE = False

# === pyFFTW 加速支持（自动回退到 NumPy）===
try:
    import pyfftw
    import pyfftw.interfaces.numpy_fft as fft
    # 启用缓存以进一步优化性能
    pyfftw.interfaces.cache.enable()
    PYFFTW_AVAILABLE = True
    print("✓ 使用 pyFFTW 加速 FFT 计算")
except ImportError:
    import numpy.fft as fft
    PYFFTW_AVAILABLE = False
    print("⚠ 使用 NumPy FFT (安装 pyfftw 可提速 2-5 倍)")

try:
    import mplcursors
    MPLCURSORS_AVAILABLE = True
except ImportError:
    MPLCURSORS_AVAILABLE = False
    print("警告: mplcursors 未安装，将无法显示交互式坐标标签")
    print("请运行: pip install mplcursors")


class AdcFFTAnalysis:
    """ADC FFT分析功能"""

    @staticmethod
    def analyze_and_plot(i_data, q_data, fs, title="", normalize=False, window_correction_mode="calibrated", window_alpha=0.5, dc_mask_width=0, window_type="hann", fund_span=15, exclude_image=False, image_span=1, power_offset_db=-0.004, noise_hann_correction=True, show_constellation=False):
        """
        分析并绘制FFT - 已校准至 adc_IQ_fft V1.3.exe
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("需要matplotlib库进行FFT分析")

        # === 1. 归一化 (校准点: 2047.0) ===
        # EXE使用 2047.0 作为满量程 (0dBFS)
        NORM_FACTOR = 2047.0
        i_normalized = i_data / NORM_FACTOR
        q_normalized = q_data / NORM_FACTOR

        # 用于时域显示的数据
        if normalize:
            i_display = i_normalized
            q_display = q_normalized
        else:
            i_display = i_data
            q_display = q_data

        # 构建复数IQ数据
        complex_data = i_normalized + 1j * q_normalized
        N = len(complex_data)

        # === 2. 加窗 (窗函数类型选择) ===
        # 使用用户选择的窗函数类型
        n = np.arange(N)
        
        if window_type == "rectangular":
            # 矩形窗（无窗）
            window = np.ones(N)
            window_name = "Rectangular (矩形窗)"
            
        elif window_type == "hann":
            # Hann窗（使用Alpha参数，默认0.5）
            alpha = window_alpha
            window = alpha - (1 - alpha) * np.cos(2 * np.pi * n / N)
            window_name = f"Hann Alpha={alpha:.3f}"
            
        elif window_type == "hamming":
            # Hamming窗（固定Alpha=0.54）
            window = 0.54 - 0.46 * np.cos(2 * np.pi * n / N)
            window_name = "Hamming (α=0.54)"
            
        elif window_type == "blackman_harris":
            # Blackman-Harris窗（4项）
            a0 = 0.35875
            a1 = 0.48829
            a2 = 0.14128
            a3 = 0.01168
            window = (a0 - a1 * np.cos(2*np.pi*n/N) + 
                     a2 * np.cos(4*np.pi*n/N) - 
                     a3 * np.cos(6*np.pi*n/N))
            window_name = "Blackman-Harris (4项)"
            
        elif window_type == "blackman":
            # Standard Blackman窗 (3项) - 匹配源程序!
            # w[n] = 0.42 - 0.5*cos(2pi*n/N) + 0.08*cos(4pi*n/N)
            window = 0.42 - 0.5 * np.cos(2 * np.pi * n / N) + 0.08 * np.cos(4 * np.pi * n / N)
            window_name = "Blackman (标准3项)"
            
        else:
            # 默认回退到Hann窗
            alpha = window_alpha
            window = alpha - (1 - alpha) * np.cos(2 * np.pi * n / N)
            window_name = f"Hann Alpha={alpha:.3f} (默认)"
        
        windowed_data = complex_data * window

        # === 3. FFT（自动使用 pyFFTW 或 NumPy）===
        fft_result = fft.fft(windowed_data)
        fft_result = fft.fftshift(fft_result)
        
        # 频率轴
        freqs_normalized = fft.fftfreq(N, 1.0)
        freqs_normalized = fft.fftshift(freqs_normalized)

        # === 4. 功率谱计算 (校准核心) ===
        
        # A. 能量校正因子 (S2) - 用于计算总功率、噪声功率 (Energy Conservation)
        # Hann窗 S2 ≈ 0.375
        S2 = np.sum(window**2) / N
        
        # B. 幅度校正因子 (CG) - 用于频谱图显示 (让0dBFS正弦波峰值为0dB)
        # Hann窗 CG ≈ 0.5
        CG = np.sum(window) / N
        
        # C. 等效噪声带宽 (ENBW) - 用于Noise/Hz计算
        # ENBW = N * S2 / (sum(window)^2) = S2 / CG^2
        # 对于Hann窗: 0.375 / 0.25 = 1.5 bins
        ENBW = S2 / (CG**2)

        # --- 计算用于显示的频谱 (幅度校正) ---
        # 这样 0dBFS 的正弦波在频谱图上峰值会正好是 0dB
        mag_spec = np.abs(fft_result) / (N * CG)
        psd_display = 20 * np.log10(mag_spec + 1e-12)

        # --- 计算用于统计的功率谱 (能量校正) ---
        # 这样 sum(psd_energy) = 时域平均功率
        psd_energy = (np.abs(fft_result) / N)**2 / S2

        # === 5. 信号参数计算 ===
        
        # 1. Total Power (总功率)
        # 使用能量谱积分
        # 修正: Total Power 始终包含 DC 能量 (用户要求)
        total_power_lin = np.sum(psd_energy)
        total_power = 10 * np.log10(total_power_lin + 1e-12) + power_offset_db

        # 2. Fund Power (基波功率)
        # 查找峰值
        if dc_mask_width >= 0:
            # 忽略 DC 附近 ±dc_mask_width bins
            # width=0 -> 屏蔽 1个点 (DC)
            # width=5 -> 屏蔽 11个点 (DC±5)
            dc_idx = N // 2
            psd_for_peak = psd_energy.copy()
            psd_for_peak[max(0, dc_idx-dc_mask_width):min(N, dc_idx+dc_mask_width+1)] = 0
            peak_idx = np.argmax(psd_for_peak)
        else:
            peak_idx = np.argmax(psd_energy)
        
        # EXE逻辑推断: 使用积分功率，积分范围可配置 (默认±15 bins)
        span = fund_span
        start_idx = max(0, peak_idx - span)
        end_idx = min(N, peak_idx + span + 1)
        
        fund_energy_lin = np.sum(psd_energy[start_idx:end_idx])
        fund_power = 10 * np.log10(fund_energy_lin + 1e-12) + power_offset_db
        
        # 基波频率
        fund_freq = freqs_normalized[peak_idx] * fs / 1e6  # MHz

        # 3. Noise Power & SNR
        # 噪声功率 = 总功率 - 基波能量 (- 谐波能量, 可选)
        # 注意：如果启用DC屏蔽，需要从噪声中减去DC能量
        noise_power_lin = total_power_lin - fund_energy_lin
        
        if dc_mask_width >= 0:
            dc_idx = N // 2
            # 计算DC能量
            dc_start = max(0, dc_idx-dc_mask_width)
            dc_end = min(N, dc_idx+dc_mask_width+1)
            dc_energy_masked = np.sum(psd_energy[dc_start:dc_end])
            
            print(f"\n[DC屏蔽调试] 屏蔽范围: {dc_mask_width} bins")
            print(f"  DC能量: {10*np.log10(dc_energy_masked+1e-12):.2f} dBFS")
            print(f"  剔除前噪声: {10*np.log10(noise_power_lin+1e-12):.2f} dBFS")
            noise_power_lin -= dc_energy_masked
            print(f"  剔除后噪声: {10*np.log10(noise_power_lin+1e-12):.2f} dBFS")
        
            print(f"  剔除后噪声: {10*np.log10(noise_power_lin+1e-12):.2f} dBFS")
        
        # 可选: 剔除镜像 (Image Removal)
        if exclude_image:
            print(f"\n[镜像剔除调试] 宽度: ±{image_span} bins")
            # 计算镜像位置
            dc_idx = N // 2
            # 基波相对DC的偏移
            fund_offset = peak_idx - dc_idx
            # 镜像位置：DC - offset
            image_idx = dc_idx - fund_offset
            
            if 0 <= image_idx < N:
                # 积分镜像能量
                img_start = max(0, image_idx - image_span)
                img_end = min(N, image_idx + image_span + 1)
                image_energy = np.sum(psd_energy[img_start:img_end])
                
                img_freq = freqs_normalized[image_idx] * fs / 1e6
                img_pwr_db = 10*np.log10(image_energy + 1e-12)
                print(f"  基波位置: bin {peak_idx}, {fund_freq:.3f} MHz")
                print(f"  镜像位置: bin {image_idx}, {img_freq:.3f} MHz")
                print(f"  镜像功率: {img_pwr_db:.2f} dBFS ({img_pwr_db - fund_power:.2f} dBc)")
                print(f"  剔除前噪声: {10*np.log10(noise_power_lin):.2f} dBFS")
                noise_power_lin -= image_energy
                print(f"  剔除后噪声: {10*np.log10(noise_power_lin):.2f} dBFS")
            else:
                print(f"  镜像位置超出范围 (bin {image_idx})")
        
        if noise_power_lin <= 1e-15: noise_power_lin = 1e-15
        
        snr = 10 * np.log10(fund_energy_lin / noise_power_lin)
        
        # 4. SNRFS
        # SNRFS = Fund Power (if 0dBFS) - Noise Floor
        # 通常定义为: 满量程信号功率 / 噪声功率
        # 如果 Fund Power 接近 0dBFS，SNRFS ≈ SNR
        # 这里我们直接用 0 - Noise Power (dB)
        noise_power_db = 10 * np.log10(noise_power_lin)
        snrfs = 0 - noise_power_db

        # 5. SFDR（无杂散动态范围）= 基波功率 - 最大杂散功率
        # 使用幅度谱(psd_display)查找最大杂散
        # 屏蔽基波附近 (校准: ±6 bins)
        psd_masked = psd_display.copy()
        mask_span = 6
        psd_masked[max(0, peak_idx-mask_span):min(N, peak_idx+mask_span+1)] = -200 # 屏蔽基波
        
        # 修正: 如果启用了DC屏蔽，SFDR计算也应该屏蔽DC
        if dc_mask_width >= 0:
            dc_idx = N // 2
            psd_masked[max(0, dc_idx-dc_mask_width):min(N, dc_idx+dc_mask_width+1)] = -200
            
        # 修正: 如果启用了镜像屏蔽，SFDR计算也应该屏蔽镜像
        if exclude_image:
            dc_idx = N // 2
            fund_offset = peak_idx - dc_idx
            image_idx = dc_idx - fund_offset
            if 0 <= image_idx < N:
                psd_masked[max(0, image_idx-image_span):min(N, image_idx+image_span+1)] = -200

        spur_peak = np.max(psd_masked)
        
        # SFDR = 杂散功率(峰值) - 基波功率(积分值)
        # EXE定义: 负数 (例如 -77dBc 表示杂散比基波低77dB)
        sfdr = spur_peak - fund_power

        # 6. Noise/Hz
        # 公式: Noise Power (dB) - 10*log10(fs) - 10*log10(ENBW)
        # 减去ENBW是为了归一化到1Hz带宽 (因为FFT bin宽度是 fs/N，且有窗函数加宽)
        # 修正: 如果启用Hann校正，强制使用Hann窗的ENBW (1.5)
        if noise_hann_correction:
            # Hann窗 ENBW = 1.5
            # 10*log10(1.5) ≈ 1.7609 dB
            enbw_val = 1.5
        else:
            enbw_val = ENBW
            
        noise_per_hz = noise_power_db - 10 * np.log10(fs) - 10 * np.log10(enbw_val)

        # 7. Channel Power (信道功率)
        # 计算固定频率范围的功率：±fs/4 (例如：40MHz采样率 -> ±10MHz)
        # 这样无论FFT点数多少，频率范围都是固定的
        center = N // 2
        # 计算±fs/4对应的bin范围
        # 频率分辨率 = fs/N，需要覆盖 fs/4 的范围
        # bin数 = (fs/4) / (fs/N) = N/4
        q_span = N // 4  # 对应 ±fs/4 的频率范围
        channel_energy_lin = np.sum(psd_energy[center-q_span:center+q_span])
        
        # 如果启用了 DC 屏蔽 (>=0)，Channel Power 也应该扣除 DC 能量
        if dc_mask_width >= 0:
            dc_idx = N // 2
            # 确保扣除范围在 Channel 带宽内
            mask_start = max(center-q_span, dc_idx-dc_mask_width)
            mask_end = min(center+q_span, dc_idx+dc_mask_width+1)
            if mask_end > mask_start:
                dc_energy_in_channel = np.sum(psd_energy[mask_start:mask_end])
                channel_energy_lin -= dc_energy_in_channel
                if channel_energy_lin < 1e-15: channel_energy_lin = 1e-15
                
        channel_power = 10 * np.log10(channel_energy_lin + 1e-12) + power_offset_db
        
        # 8. DC Power (直流功率)
        # DC bin 位于频谱中心
        dc_idx = N // 2
        dc_power_lin = psd_energy[dc_idx]
        dc_power = 10 * np.log10(dc_power_lin + 1e-12) + power_offset_db

        # 9. Average Bin Noise (平均Bin噪声)
        # 定义: 噪声功率 / 有效FFT bin数
        # 有效bin数 = N (所有频率bin)
        avg_bin_noise_lin = noise_power_lin / N
        avg_bin_noise = 10 * np.log10(avg_bin_noise_lin + 1e-12)

        # 调试输出
        print(f"\n=== Calibrated Analysis ===")
        print(f"Fund Freq  : {fund_freq:.3f} MHz")
        print(f"Total Power: {total_power:.4f} dBFS")
        print(f"Fund Power : {fund_power:.4f} dBFS")
        print(f"SNR        : {snr:.4f} dB")
        print(f"SFDR       : {sfdr:.4f} dBc")
        print(f"Noise/Hz   : {noise_per_hz:.4f} dBFS/Hz")
        print(f"Avg Bin Noise: {avg_bin_noise:.4f} dBFS/bin")
        print(f"DC Power   : {dc_power:.4f} dBFS")
        
        print(f"\n[Noise/Hz Debug Detail]")
        print(f"1. Total Power (Lin): {total_power_lin:.6e}")
        print(f"2. Fund Power (Lin) : {fund_energy_lin:.6e} (Span: ±{span} bins)")
        print(f"3. Noise Power (Lin): {noise_power_lin:.6e} (= Total - Fund)")
        print(f"   Noise Power (dB) : {noise_power_db:.4f} dBFS")
        print(f"4. fs               : {fs:.1f} Hz")
        print(f"   10*log10(fs)     : {10*np.log10(fs):.4f} dB")
        print(f"5. Window           : {window_name}")
        print(f"   S2 (Energy)      : {S2:.6f}")
        print(f"   CG (Amplitude)   : {CG:.6f}")
        print(f"   ENBW (bins)      : {ENBW:.4f} (= S2/CG^2)")
        print(f"   10*log10(ENBW)   : {10*np.log10(ENBW):.4f} dB")
        print(f"6. Calculation      : {noise_power_db:.4f} - {10*np.log10(fs):.4f} - {10*np.log10(ENBW):.4f} = {noise_per_hz:.4f}")
        print(f"===========================\n")

        # 创建图表 -        # 设置绘图大小 (约 1100x700 像素 -> 11x7 英寸)
        if show_constellation:
            # 3个子图，稍微高一点以容纳内容，但保持紧凑
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 9))
        else:
            # 2个子图，标准大小
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7))
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # === 时域图 ===
        ax1.plot(i_display, label='I', alpha=0.7, linewidth=1, color='blue')
        ax1.plot(q_display, label='Q', alpha=0.7, linewidth=1, color='red')

        # === 参数显示优化 ===
        # 第一行: Fund Freq, Fund Power, Total Power, Channel Pwr
        # 第二行: SNR, SNRFS, Noise/Hz
        params_text = (
            f"Fund Freq={fund_freq:.2f}MHz  "
            f"Fund Power={fund_power:.3f}dBFS  "
            f"Total Power={total_power:.3f}dBFS  "
            f"Channel Pwr={channel_power:.2f}dBFS\n"
            f"SNR={snr:.2f}dB  "
            f"SNRFS={snrfs:.2f}dB  "
            f"Noise/Hz={noise_per_hz:.2f}dBFS/Hz"
        )

        if normalize:
            time_title = f'时域IQ信号\n{params_text}'
        else:
            time_title = f'时域IQ信号 (共{len(i_display)}个样本)\n{params_text}'
            
        ax1.set_title(time_title, fontsize=10, pad=15)
        ax1.set_ylabel('幅度') # 简化纵坐标标签

        # === 时域纵坐标自适应 (数据占90%) ===
        # 计算数据的实际范围
        data_min = min(np.min(i_display), np.min(q_display))
        data_max = max(np.max(i_display), np.max(q_display))
        
        if data_max == data_min:
            data_max += 0.5
            data_min -= 0.5
            
        # 计算中心和跨度
        center = (data_max + data_min) / 2
        span = data_max - data_min
        
        # 扩展跨度以留出空白 (数据占90% -> 总跨度 = span / 0.9)
        new_span = span / 0.9
        
        y_min = center - new_span / 2
        y_max = center + new_span / 2
        
        ax1.set_ylim(y_min, y_max)
        
        ax1.set_xlabel('样本')
        ax1.set_xlim(0, len(i_display) - 1)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # === 频域图 ===
        freqs_display = freqs_normalized * fs / 1e6
        
        # 显示幅度校正后的频谱 (0dBFS Sine -> 0dB Peak)
        ax2.plot(freqs_display, psd_display, linewidth=1, color='blue')

        pink_range = fs / 1e6 / 4
        ax2.axvspan(-pink_range, pink_range, alpha=0.15, color='pink')

        ax2.set_title('频谱', fontsize=12)
        ax2.set_ylabel('功率 (dBFS)')
        ax2.set_xlabel('频率 (MHz)')
        ax2.set_xlim(freqs_display[0], freqs_display[-1])
        # 固定Y轴范围: 0到-140dBFS, 每20dB一个刻度
        ax2.set_ylim(-140, 0)
        ax2.set_yticks(np.arange(-140, 1, 20))
        ax2.grid(True, alpha=0.3)

        # === IQ星座图 (可选) ===
        if show_constellation:
            if len(i_display) > 50000:
                step = len(i_display) // 50000
                i_plot = i_display[::step]
                q_plot = q_display[::step]
            else:
                i_plot = i_display
                q_plot = q_display

            ax3.scatter(i_plot, q_plot, alpha=0.3, s=1, c='blue')
            ax3.set_title(f'IQ星座图 (显示{len(i_plot)}个点)', fontsize=12)
            ax3.set_xlabel('I')
            ax3.set_ylabel('Q')
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
            ax3.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)

            if normalize:
                ax3.set_xlim(-1.2, 1.2)
                ax3.set_ylim(-1.2, 1.2)

            ax3.axis('equal')

        plt.tight_layout()

        # === 交互功能 ===
        try:
            from matplotlib.backend_bases import cursors
            fig.canvas.set_cursor(cursors.POINTER)
        except:
            pass

        if MPLCURSORS_AVAILABLE:
            lines1 = ax1.get_lines()
            if lines1:
                cursor_time = mplcursors.cursor(lines1, hover=False, multiple=True, highlight=False)
                cursor_time.connect("add", lambda sel: sel.annotation.set_text(
                    f'样本: {sel.target[0]:.0f}\n幅度: {sel.target[1]:.4f}'
                ))

            lines2 = ax2.get_lines()
            if lines2:
                cursor_freq = mplcursors.cursor(lines2, hover=False, multiple=True, highlight=False)
                cursor_freq.connect("add", lambda sel: sel.annotation.set_text(
                    f'频率: {sel.target[0]:.2f} MHz\n功率: {sel.target[1]:.2f} dBFS'
                ))

            fig.text(0.5, 0.01,
                    '[提示] 左键双击曲线附近添加标签 | 右键双击标签删除 | d键删除所有标签 | e键启用/禁用 | Home键恢复视图',
                    transform=fig.transFigure, fontsize=8, ha='center', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='gray', linewidth=0.5))

        plt.show()

        # 计算统计值 (始终使用归一化后的数据)
        power = np.mean(np.abs(complex_data)**2)
        peak_power = np.max(np.abs(complex_data)**2)

        if normalize:
            stats_i = i_normalized
            stats_q = q_normalized
        else:
            stats_i = i_data
            stats_q = q_data

        return {
            'power': 10 * np.log10(power + 1e-12),
            'peak_power': 10 * np.log10(peak_power + 1e-12),
            'papr': 10 * np.log10(peak_power + 1e-12) - 10 * np.log10(power + 1e-12),
            'rms': np.sqrt(power),
            'i_mean': np.mean(stats_i),
            'q_mean': np.mean(stats_q),
            'i_std': np.std(stats_i),
            'q_std': np.std(stats_q),
            'i_min': np.min(stats_i),
            'i_max': np.max(stats_i),
            'q_min': np.min(stats_q),
            'q_max': np.max(stats_q),
            # 新增 FFT 指标
            'fund_freq': fund_freq,
            'fund_power': fund_power,
            'total_power': total_power,
            'channel_power': channel_power,
            'snr': snr,
            'snrfs': snrfs,
            'noise_per_hz': noise_per_hz
        }


class MainWindow(QMainWindow):
    """主窗口类 - FFT Tool V1.3"""

    def __init__(self, parent=None):
        """初始化"""
        super(MainWindow, self).__init__(parent)

        self.setWindowTitle("ADC IQ FFT Tool V1.2")
        
        # 设置窗口图标 - 兼容打包环境
        icon_name = 'Gemini_Generated_Image_bud0ylbud0ylbud0.ico'
        if hasattr(sys, '_MEIPASS'):
            # PyInstaller 打包后的临时目录
            icon_path = os.path.join(sys._MEIPASS, icon_name)
        else:
            # 开发环境
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), icon_name)
            
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.filename = ""
        self.raw_data = []
        self.is_first_load = True  # 标志位：是否是首次加载

        self.setupUi()
        
        # === 窗口大小设置 ===
        self.resize(1100, 700)
        
        # 启用拖放功能
        self.setAcceptDrops(True)

    def setupUi(self):
        """设置UI界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        # === 文件选择 ===
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("文件:"))
        self.edtFilename = QLineEdit()
        self.edtFilename.setReadOnly(True)
        file_layout.addWidget(self.edtFilename)
        self.pbLoad = QPushButton("加载文件")
        self.pbLoad.clicked.connect(self.loadAndRun)
        self.pbLoad.setMinimumWidth(100)
        file_layout.addWidget(self.pbLoad)
        layout.addLayout(file_layout)

        # === 参数设置 ===
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("采样率(MHz):"))
        self.edtFs = QLineEdit("40")  # 默认40MHz，仅用于频谱横坐标显示
        self.edtFs.setMaximumWidth(120)
        param_layout.addWidget(self.edtFs)

        param_layout.addWidget(QLabel("位数:"))
        self.edtBits = QLineEdit("12")
        self.edtBits.setMaximumWidth(60)
        param_layout.addWidget(self.edtBits)

        param_layout.addStretch()
        layout.addLayout(param_layout)

        # === 路径和数据选择 ===
        select_layout = QHBoxLayout()

        select_layout.addWidget(QLabel("路径选择:"))
        self.cboxPathSel = QComboBox()
        self.cboxPathSel.addItems(["Path0", "Path1"])
        self.cboxPathSel.setMaximumWidth(120)
        select_layout.addWidget(self.cboxPathSel)

        select_layout.addWidget(QLabel("数据选择:"))
        self.cboxDataSel = QComboBox()
        self.cboxDataSel.addItems(["I&Q", "I only", "Q only"])
        self.cboxDataSel.setMaximumWidth(120)
        select_layout.addWidget(self.cboxDataSel)

        select_layout.addStretch()
        layout.addLayout(select_layout)

        # === 数据范围 (基本参数) ===
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("数据起始:"))
        self.edtdataFrom = QLineEdit("0")
        self.edtdataFrom.setMaximumWidth(100)
        range_layout.addWidget(self.edtdataFrom)

        range_layout.addWidget(QLabel("数据结束:"))
        self.edtdataTo = QLineEdit("25000")  # 默认25000个IQ点
        self.edtdataTo.setMaximumWidth(100)
        range_layout.addWidget(self.edtdataTo)

        self.cbIsDataCut = QCheckBox("裁剪数据")
        self.cbIsDataCut.setChecked(True)
        range_layout.addWidget(self.cbIsDataCut)

        self.cbIQswap = QCheckBox("IQ交换")
        range_layout.addWidget(self.cbIQswap)

        range_layout.addStretch()
        layout.addLayout(range_layout)

        # === 高级参数 (可折叠) ===
        advanced_group = QGroupBox("高级参数")
        advanced_group.setCheckable(True)
        advanced_group.setChecked(False)  # 默认折叠
        advanced_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #ccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        advanced_layout = QVBoxLayout()
        
        # === 归一化和星座图选项 ===
        norm_layout = QHBoxLayout()
        self.cbNormalize = QCheckBox("时域归一化显示 (÷2047, FFT始终用归一化数据)")
        self.cbNormalize.setChecked(True)  # 默认勾选
        norm_layout.addWidget(self.cbNormalize)
        
        self.cbShowConstellation = QCheckBox("显示星座图")
        self.cbShowConstellation.setChecked(False)  # 默认不勾选
        self.cbShowConstellation.setToolTip("勾选后显示IQ星座图，默认只显示时域图和频谱图")
        norm_layout.addWidget(self.cbShowConstellation)
        
        norm_layout.addStretch()
        advanced_layout.addLayout(norm_layout)

        # === 窗函数校正选项 ===
        window_layout = QHBoxLayout()
        window_layout.addWidget(QLabel("窗函数校正:"))
        self.cboxWindowCorr = QComboBox()
        self.cboxWindowCorr.addItems([
            "不校正 (原始FFT/N)",
            "幅度校正 (单频信号)",
            "能量校正 (噪声信号)"
        ])
        self.cboxWindowCorr.setCurrentIndex(0)  # 默认选择"不校正"
        self.cboxWindowCorr.setMaximumWidth(200)
        window_layout.addWidget(self.cboxWindowCorr)
        
        # === 窗函数类型选择 ===
        window_layout.addWidget(QLabel("窗类型:"))
        self.cboxWindowType = QComboBox()
        self.cboxWindowType.addItems([
            "Hann (Alpha窗)",
            "Rectangular (矩形窗)",
            "Hamming (汉明窗)",
            "Blackman-Harris",
            "Blackman (标准3项)"
        ])
        self.cboxWindowType.setCurrentIndex(4)  # 默认选择Blackman窗 (匹配源程序)
        self.cboxWindowType.setMaximumWidth(160)
        self.cboxWindowType.setToolTip("窗函数类型选择:\\n"
                                       "Hann - 默认，使用Alpha参数\\n"
                                       "Rectangular - 无窗，所有点权重相同\\n"
                                       "Hamming - 固定α=0.54\\n"
                                       "Blackman-Harris - 高SFDR性能\\n"
                                       "Blackman - 标准3项窗 (匹配源程序)")
        window_layout.addWidget(self.cboxWindowType)
        
        # === 自适应窗函数选项 (已移除) ===
        # self.cbAdaptiveWindow = QCheckBox("自适应窗函数 (单音信号优化)")
        # self.cbAdaptiveWindow.setChecked(False)  # 默认不勾选
        # window_layout.addWidget(self.cbAdaptiveWindow)
        
        # === 谐波减除选项 (已移除) ===
        # self.cbSubtractHarmonics = QCheckBox("谐波减除 (单音信号SNR优化)")
        # self.cbSubtractHarmonics.setChecked(False)  # 默认不勾选
        # window_layout.addWidget(self.cbSubtractHarmonics)
        
        # === DC屏蔽范围 ===
        
        # === DC屏蔽范围 ===
        window_layout.addWidget(QLabel("DC屏蔽:"))
        self.edtDCMask = QLineEdit("2")  # 默认屏蔽DC±2 (5个点)
        self.edtDCMask.setMaximumWidth(40)
        self.edtDCMask.setToolTip("DC屏蔽范围 (bins):\\n -1 = 不屏蔽\\n  0 = 仅屏蔽DC (1个点)\\n  2 = 屏蔽DC±2点 (5个点, 默认)\\n  5 = 屏蔽DC±5点 (11个点)")
        window_layout.addWidget(self.edtDCMask)
        
        # === Fund积分范围 ===
        window_layout.addWidget(QLabel("Fund积分:"))
        self.edtFundSpan = QLineEdit("10")  # 默认10 bins (校准值)
        self.edtFundSpan.setMaximumWidth(40)
        self.edtFundSpan.setToolTip("Fund Power积分范围 (±bins):\\n"
                                    "10 = 默认 (校准值)\\n"
                                    "5 = 窄范围 (适合尖锐峰值)\\n"
                                    "15 = 宽范围 (适合Hann窗)")
        window_layout.addWidget(self.edtFundSpan)
        
        window_layout.addStretch()
        advanced_layout.addLayout(window_layout)

        # === 杂散剔除选项 (已移除) ===
        # spur_layout = QHBoxLayout()
        # self.cbExcludeSpurs = QCheckBox("剔除杂散 (Spur Removal)")
        # spur_layout.addWidget(self.cbExcludeSpurs)
        # spur_layout.addWidget(QLabel("阈值(dBc):"))
        # self.edtSpurThresh = QLineEdit("-60")
        # spur_layout.addWidget(self.edtSpurThresh)
        # spur_layout.addWidget(QLabel("宽度(bins):"))
        # self.edtSpurSpan = QLineEdit("1")
        # spur_layout.addWidget(self.edtSpurSpan)
        # layout.addLayout(spur_layout)

        # === 镜像剔除选项 ===
        image_layout = QHBoxLayout()
        self.cbExcludeImage = QCheckBox("剔除镜像 (Image Removal)")
        self.cbExcludeImage.setChecked(True) # 默认开启
        self.cbExcludeImage.setToolTip("启用后，自动剔除基波的对称镜像信号能量\\n(用于IQ不平衡信号的SNR优化)")
        image_layout.addWidget(self.cbExcludeImage)
        
        image_layout.addWidget(QLabel("宽度(bins):"))
        self.edtImageSpan = QLineEdit("1")
        self.edtImageSpan.setMaximumWidth(30)
        self.edtImageSpan.setToolTip("剔除镜像时的积分宽度 (±bins)")
        image_layout.addWidget(self.edtImageSpan)
        
        # === 校准选项 ===
        image_layout.addStretch()
        image_layout.addWidget(QLabel("Power Offset(dB):"))
        self.edtPowerOffset = QLineEdit("-0.004")
        self.edtPowerOffset.setMaximumWidth(60)
        image_layout.addWidget(self.edtPowerOffset)
        self.edtPowerOffset.setToolTip("全局功率偏移校准 (dB)\\n应用于 Fund/Total/DC/Channel Power")
        
        self.cbNoiseHannCorr = QCheckBox("Noise/Hz Hann校正")
        self.cbNoiseHannCorr.setChecked(True)
        self.cbNoiseHannCorr.setToolTip("启用后，Noise/Hz计算始终使用Hann窗的ENBW (1.5 bins)\\n用于消除与源程序(可能固定使用Hann参数)的0.61dB差异")
        image_layout.addWidget(self.cbNoiseHannCorr)
        
        advanced_layout.addLayout(image_layout)
        
        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)

        # === 说明 ===
        info_label = QLabel("数据排布: Path0_I, Path0_Q, Path1_I, Path1_Q (每4行循环) | 数据范围以IQ点数为单位 | 采样率仅用于频谱横坐标显示")
        info_label.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(info_label)

        # === 运行按钮 ===
        self.pbUpdate = QPushButton("运行FFT分析")
        self.pbUpdate.clicked.connect(self.getdataAndRun)
        self.pbUpdate.setMinimumHeight(45)
        self.pbUpdate.setStyleSheet("""
            QPushButton {
                font-size: 13pt;
                font-weight: bold;
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        layout.addWidget(self.pbUpdate)

        # === 数据预览 ===
        layout.addWidget(QLabel("数据预览:"))
        self.txtData = QTextEdit()
        self.txtData.setMaximumHeight(150)
        self.txtData.setReadOnly(True)
        self.txtData.setStyleSheet("font-family: Consolas, Monaco, monospace; font-size: 9pt; background-color: #f5f5f5;")
        layout.addWidget(self.txtData)

        # === 消息日志 ===
        layout.addWidget(QLabel("消息日志:"))
        self.lblMsg = QTextEdit()
        self.lblMsg.setMaximumHeight(120)
        self.lblMsg.setReadOnly(True)
        self.lblMsg.setStyleSheet("background-color: #f9f9f9;")
        layout.addWidget(self.lblMsg)

    @pyqtSlot()
    def loadAndRun(self):
        """选择并加载文件"""
        filename, ok = QFileDialog.getOpenFileName(
            self,
            "打开IQ数据文件",
            "",
            "TXT File(*.txt);;All Files (*.*)"
        )

        if not ok or not filename:
            return

        self.filename = filename
        self.edtFilename.setText(filename)
        self.loadFile()
    
    def loadFile(self):
        """加载文件数据"""
        if not self.filename:
            self.logPrint("✗ 错误: 未指定文件", error=True)
            return

        try:
            with open(self.filename, 'r') as f:
                lines = f.readlines()

            # 读取原始hex值
            raw_hex_values = []
            for line in lines:
                line = line.strip()
                if line and line.startswith('0x'):
                    try:
                        raw_hex_values.append(line)
                    except:
                        continue

            if len(raw_hex_values) == 0:
                raise ValueError("未找到有效的hex数据")

            # 检测格式：检查第一行的hex字符串长度（去除'0x'）
            first_hex = raw_hex_values[0]
            hex_length = len(first_hex) - 2  # 去除'0x'前缀
            
            if hex_length == 8:
                # 32bit格式：每行包含2个12bit数据
                format_type = "32bit"
                self.logPrint(f"✓ 检测到32bit打包格式 (0x + 8位hex)")
                
                # 解包32bit数据为16bit数据
                unpacked_data = []
                for hex_str in raw_hex_values:
                    val32 = int(hex_str, 16)
                    
                    # 提取低12bit和中12bit
                    low12 = val32 & 0xFFF
                    mid12 = (val32 >> 12) & 0xFFF
                    
                    # 转换为有符号数
                    if low12 & 0x800:  # 符号位为1
                        low12 = low12 - 0x1000
                    if mid12 & 0x800:
                        mid12 = mid12 - 0x1000
                    
                    unpacked_data.append(low12)
                    unpacked_data.append(mid12)
                
                self.raw_data = unpacked_data
                self.logPrint(f"✓ 32bit格式解包: {len(raw_hex_values)}行 -> {len(self.raw_data)}个16bit数据")
                
            elif hex_length == 4:
                # 16bit格式：每行1个12bit数据
                format_type = "16bit"
                self.logPrint(f"✓ 检测到16bit标准格式 (0x + 4位hex)")
                
                # 转换为有符号数
                self.raw_data = []
                bits = int(self.edtBits.text())
                for hex_str in raw_hex_values:
                    val = int(hex_str, 16)
                    if val & (1 << (bits-1)):
                        val = val - (1 << bits)
                    self.raw_data.append(val)
            else:
                raise ValueError(f"未知的hex格式长度: {hex_length} (期望4或8)")

            # 计算IQ点数（4行原始数据 = 1个IQ点）
            if len(self.raw_data) % 4 != 0:
                self.logPrint(f"⚠ 警告: 数据点数不是4的倍数", error=False)

            num_iq_points = len(self.raw_data) // 4

            self.logPrint(f"✓ 文件加载成功: {os.path.basename(self.filename)}")
            self.logPrint(f"✓ 原始数据行数: {len(self.raw_data)}")
            self.logPrint(f"✓ IQ数据点数: {num_iq_points} (每个点4行)")

            # 显示前几组（以IQ点为单位）
            txt_lines = []
            txt_lines.append(f"格式: {format_type}")
            txt_lines.append("IQ点   Path0_I  Path0_Q  Path1_I  Path1_Q")
            txt_lines.append("-" * 60)

            display_points = min(8, num_iq_points)  # 显示前8个IQ点
            for iq_idx in range(display_points):
                base_idx = iq_idx * 4
                p0_i = self.raw_data[base_idx]
                p0_q = self.raw_data[base_idx + 1]
                p1_i = self.raw_data[base_idx + 2]
                p1_q = self.raw_data[base_idx + 3]

                txt_lines.append(f"[{iq_idx:4d}] {p0_i:6d}  {p0_q:6d}  {p1_i:6d}  {p1_q:6d}")

            if num_iq_points > display_points:
                txt_lines.append(f"\n... (共 {num_iq_points} 个IQ点)")

            self.txtData.setText('\n'.join(txt_lines))

            # 自动填入总IQ点数到"数据结束"框
            self.edtdataTo.setText(str(num_iq_points))

            # === 自动运行逻辑 ===
            if self.is_first_load:
                self.is_first_load = False
                self.logPrint("\n[提示] 首次加载，请配置参数后点击'运行分析'")
            else:
                self.logPrint("\n[自动运行] 检测到新文件，自动执行分析...")
                # 使用 QTimer.singleShot 稍微延迟执行，确保UI刷新
                from PyQt5.QtCore import QTimer
                QTimer.singleShot(100, self.getdataAndRun)

        except Exception as e:
            self.logPrint(f"✗ 错误: {str(e)}", error=True)
            QMessageBox.critical(self, "错误", f"加载文件失败:\n{str(e)}")

    @pyqtSlot()
    def getdataAndRun(self):
        """执行FFT分析"""
        if not self.raw_data:
            self.logPrint("✗ 错误: 请先加载数据文件", error=True)
            QMessageBox.warning(self, "警告", "请先加载数据文件")
            return

        try:
            # 获取参数
            fs_mhz = float(self.edtFs.text())  # MHz
            fs = fs_mhz * 1e6  # 转换为Hz用于内部计算（虽然现在不影响FFT，只用于显示）
            bits = int(self.edtBits.text())
            path_sel = self.cboxPathSel.currentText()
            data_sel_mode = self.cboxDataSel.currentText()
            iqswap = self.cbIQswap.isChecked()
            normalize = self.cbNormalize.isChecked()
            iqswap = self.cbIQswap.isChecked()
            normalize = self.cbNormalize.isChecked()
            
            # 窗函数校正模式
            window_corr_index = self.cboxWindowCorr.currentIndex()
            window_corr_modes = ["none", "amplitude", "energy"]
            window_corr_mode = window_corr_modes[window_corr_index]
            
            
            # 窗函数Alpha参数 (固定值，仅用于Hann窗)
            window_alpha = 0.5
            
            # 窗函数类型
            window_type_index = self.cboxWindowType.currentIndex()
            window_type_map = ["hann", "rectangular", "hamming", "blackman_harris", "blackman"]
            window_type = window_type_map[window_type_index]

            # 数据范围（用户输入的是IQ点索引，需要转换为原始行索引）
            num_iq_points = len(self.raw_data) // 4
            iq_start = 0
            iq_stop = num_iq_points

            if self.cbIsDataCut.isChecked():
                iq_start = int(self.edtdataFrom.text())
                iq_stop = int(self.edtdataTo.text())

            if iq_start < 0 or iq_stop > num_iq_points or iq_start >= iq_stop:
                raise ValueError(f"IQ点范围无效: {iq_start} ~ {iq_stop}，总IQ点数: {num_iq_points}")

            # 转换为原始行索引（1个IQ点 = 4行原始数据）
            dataRowstart = iq_start * 4
            dataRowstop = iq_stop * 4

            data_subset = self.raw_data[dataRowstart:dataRowstop]

            # 分离Path数据 (每4行: Path0_I, Path0_Q, Path1_I, Path1_Q)
            path0_i = np.array(data_subset[0::4], dtype=np.float64)
            path0_q = np.array(data_subset[1::4], dtype=np.float64)
            path1_i = np.array(data_subset[2::4], dtype=np.float64)
            path1_q = np.array(data_subset[3::4], dtype=np.float64)

            # 选择Path
            if path_sel == "Path0":
                selected_i, selected_q = path0_i, path0_q
            else:
                selected_i, selected_q = path1_i, path1_q

            # IQ交换
            if iqswap:
                selected_i, selected_q = selected_q, selected_i

            # 数据模式
            if data_sel_mode == "I only":
                final_i, final_q = selected_i, np.zeros_like(selected_i)
                mode_desc = "I only"
            elif data_sel_mode == "Q only":
                final_i, final_q = np.zeros_like(selected_q), selected_q
                mode_desc = "Q only"
            else:
                final_i, final_q = selected_i, selected_q
                mode_desc = "I&Q"

            self.logPrint(f"\n{'='*70}")
            self.logPrint(f"数据范围:")
            self.logPrint(f"  文件总原始行数: {len(self.raw_data)}")
            self.logPrint(f"  文件总IQ点数: {num_iq_points}")
            self.logPrint(f"  用户选择IQ点范围: [{iq_start}:{iq_stop}] (共{iq_stop-iq_start}个IQ点)")
            self.logPrint(f"  对应原始行范围: [{dataRowstart}:{dataRowstop}] (共{dataRowstop-dataRowstart}行)")
            self.logPrint(f"  每个Path的IQ样本数: {len(path0_i)}")

            self.logPrint(f"\n分析参数:")
            self.logPrint(f"  路径: {path_sel}")
            self.logPrint(f"  数据模式: {mode_desc}")
            self.logPrint(f"  采样率显示: {fs_mhz:.2f} MHz (仅用于频谱横坐标缩放)")
            self.logPrint(f"  归一化显示: {'是 (÷2047)' if normalize else '否'}")
            # DC屏蔽范围
            try:
                dc_mask_width = int(self.edtDCMask.text())
                # -1 表示不屏蔽，>=0 表示屏蔽DC±width bins
            except ValueError:
                dc_mask_width = -1  # 默认不屏蔽
                self.edtDCMask.setText("-1")
            
            # Fund积分范围
            try:
                fund_span = int(self.edtFundSpan.text())
                if fund_span < 1:
                    fund_span = 15
                    self.edtFundSpan.setText("15")
            except ValueError:
                fund_span = 15
                self.edtFundSpan.setText("15")
            
            self.logPrint(f"  窗函数校正: {self.cboxWindowCorr.currentText()}")
            self.logPrint(f"  窗函数类型: {self.cboxWindowType.currentText()}")
            self.logPrint(f"  窗Alpha参数: {window_alpha}")
            self.logPrint(f"  DC屏蔽范围: ±{dc_mask_width} bins")
            self.logPrint(f"  Fund积分范围: ±{fund_span} bins")
            self.logPrint(f"  最终IQ样本数: {len(final_i)}")
            self.logPrint(f"  时域图显示: 全部{len(final_i)}个样本")

            # 杂散剔除参数 (已移除)
            # exclude_spurs = self.cbExcludeSpurs.isChecked()
            # ...
                
            # 镜像剔除参数
            exclude_image = self.cbExcludeImage.isChecked()
            try:
                image_span = int(self.edtImageSpan.text())
                if image_span < 0: image_span = 0
            except ValueError:
                image_span = 1
                self.edtImageSpan.setText("1")
                
            # 校准参数
            try:
                power_offset_db = float(self.edtPowerOffset.text())
            except ValueError:
                power_offset_db = -0.004
                self.edtPowerOffset.setText("-0.004")
                
            noise_hann_correction = self.cbNoiseHannCorr.isChecked()
            
            # 星座图显示选项
            show_constellation = self.cbShowConstellation.isChecked()

            # FFT分析
            self.logPrint(f"\n开始FFT分析...")
            analyzer = AdcFFTAnalysis()
            chart_title = f"{path_sel} - {mode_desc} - {os.path.basename(self.filename)}"
            result = analyzer.analyze_and_plot(final_i, final_q, fs, chart_title, normalize, window_corr_mode, window_alpha, dc_mask_width, window_type, fund_span, exclude_image, image_span, power_offset_db, noise_hann_correction, show_constellation)

            # 显示结果
            self.logPrint(f"\n✓ 分析完成!")
            self.logPrint(f"{'='*70}")
            
            # 打印 FFT 关键指标 (与绘图标题一致)
            self.logPrint(f"FFT 分析结果:")
            self.logPrint(f"  Fund Freq   : {result['fund_freq']:.2f} MHz")
            self.logPrint(f"  Fund Power  : {result['fund_power']:.3f} dBFS")
            self.logPrint(f"  Total Power : {result['total_power']:.3f} dBFS")
            self.logPrint(f"  Channel Pwr : {result['channel_power']:.2f} dBFS")
            self.logPrint(f"  SNR         : {result['snr']:.2f} dB")
            self.logPrint(f"  SNRFS       : {result['snrfs']:.2f} dB")
            self.logPrint(f"  Noise/Hz    : {result['noise_per_hz']:.2f} dBFS/Hz")
            self.logPrint(f"{'-'*30}")
            
            self.logPrint(f"信号统计 ({'归一化' if normalize else '原始值'}):")
            self.logPrint(f"  峰值功率: {result['peak_power']:.2f} dB")
            self.logPrint(f"  峰均比(PAPR): {result['papr']:.2f} dB")
            self.logPrint(f"  RMS: {result['rms']:.4f}")

            if normalize:
                self.logPrint(f"  I均值: {result['i_mean']:+.6f}")
                self.logPrint(f"  Q均值: {result['q_mean']:+.6f}")
                self.logPrint(f"  I标准差: {result['i_std']:.6f}")
                self.logPrint(f"  Q标准差: {result['q_std']:.6f}")
                self.logPrint(f"  I范围: [{result['i_min']:+.6f}, {result['i_max']:+.6f}]")
                self.logPrint(f"  Q范围: [{result['q_min']:+.6f}, {result['q_max']:+.6f}]")
            else:
                self.logPrint(f"  I均值: {result['i_mean']:.2f}")
                self.logPrint(f"  Q均值: {result['q_mean']:.2f}")
                self.logPrint(f"  I标准差: {result['i_std']:.2f}")
                self.logPrint(f"  Q标准差: {result['q_std']:.2f}")
                self.logPrint(f"  I范围: [{result['i_min']:.0f}, {result['i_max']:.0f}]")
                self.logPrint(f"  Q范围: [{result['q_min']:.0f}, {result['q_max']:.0f}]")
            self.logPrint(f"{'='*70}\n")

            QApplication.processEvents()

        except Exception as e:
            import traceback
            self.logPrint(f"\n✗ 分析错误: {str(e)}", error=True)
            self.logPrint(traceback.format_exc(), error=True)
            QMessageBox.critical(self, "错误", f"分析失败:\n{str(e)}")

    def dragEnterEvent(self, event):
        """处理拖动进入事件"""
        if event.mimeData().hasUrls():
            # 检查是否有文件被拖入
            urls = event.mimeData().urls()
            if len(urls) == 1 and urls[0].isLocalFile():
                # 只接受单个本地文件
                file_path = urls[0].toLocalFile()
                if file_path.lower().endswith('.txt'):
                    event.acceptProposedAction()
                    return
        event.ignore()
    
    def dropEvent(self, event):
        """处理文件拖放事件"""
        urls = event.mimeData().urls()
        if len(urls) == 1 and urls[0].isLocalFile():
            file_path = urls[0].toLocalFile()
            if file_path.lower().endswith('.txt'):
                # 设置文件路径并自动加载
                self.filename = file_path
                self.edtFilename.setText(file_path)
                self.logPrint(f"\n📁 拖放文件: {os.path.basename(file_path)}")
                # 调用加载函数
                self.loadFile()
                event.acceptProposedAction()
            else:
                self.logPrint("✗ 错误: 只支持 .txt 文件", error=True)
                event.ignore()
        else:
            self.logPrint("✗ 错误: 一次只能拖放一个文件", error=True)
            event.ignore()
    
    def logPrint(self, msg, error=False):
        """打印日志"""
        if error:
            self.lblMsg.setStyleSheet("QTextEdit { color: red; background-color: #fff5f5; }")
        else:
            self.lblMsg.setStyleSheet("QTextEdit { background-color: #f9f9f9; }")

        self.lblMsg.append(msg)
        self.lblMsg.moveCursor(QTextCursor.End)
        QApplication.processEvents()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

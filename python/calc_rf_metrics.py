from typing import List

import numpy as np
import numpy.fft as fft



def get_rf_metrics(i_data: List[int], q_data: List[int], fs: int):
    power_offset_db = -0.004
    dc_mask_width = 2
    fund_span = 10
    exclude_image = True
    image_span = 1
    noise_hann_correction = True

    i_data = np.array(i_data, dtype=np.float64)
    q_data = np.array(q_data, dtype=np.float64)
    fs = fs * 1e6

    NORM_FACTOR = 2047.0
    i_normalized = i_data / NORM_FACTOR
    q_normalized = q_data / NORM_FACTOR

    complex_data = i_normalized + 1j * q_normalized
    N = len(complex_data)

    # === 2. 加窗 (窗函数类型选择) ===
    # 使用用户选择的窗函数类型
    n = np.arange(N)

    window = 0.42 - 0.5 * np.cos(2 * np.pi * n / N) + 0.08 * np.cos(4 * np.pi * n / N)
    window_name = "Blackman (标准3项)"

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
    S2 = np.sum(window ** 2) / N

    # B. 幅度校正因子 (CG) - 用于频谱图显示 (让0dBFS正弦波峰值为0dB)
    # Hann窗 CG ≈ 0.5
    CG = np.sum(window) / N

    # C. 等效噪声带宽 (ENBW) - 用于Noise/Hz计算
    # ENBW = N * S2 / (sum(window)^2) = S2 / CG^2
    # 对于Hann窗: 0.375 / 0.25 = 1.5 bins
    ENBW = S2 / (CG ** 2)

    # --- 计算用于显示的频谱 (幅度校正) ---
    # 这样 0dBFS 的正弦波在频谱图上峰值会正好是 0dB
    mag_spec = np.abs(fft_result) / (N * CG)
    psd_display = 20 * np.log10(mag_spec + 1e-12)

    # --- 计算用于统计的功率谱 (能量校正) ---
    # 这样 sum(psd_energy) = 时域平均功率
    psd_energy = (np.abs(fft_result) / N) ** 2 / S2

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
        psd_for_peak[max(0, dc_idx - dc_mask_width):min(N, dc_idx + dc_mask_width + 1)] = 0
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
        dc_start = max(0, dc_idx - dc_mask_width)
        dc_end = min(N, dc_idx + dc_mask_width + 1)
        dc_energy_masked = np.sum(psd_energy[dc_start:dc_end])

        noise_power_lin -= dc_energy_masked

    # 可选: 剔除镜像 (Image Removal)
    if exclude_image:
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
            img_pwr_db = 10 * np.log10(image_energy + 1e-12)
            noise_power_lin -= image_energy

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
    psd_masked[max(0, peak_idx - mask_span):min(N, peak_idx + mask_span + 1)] = -200  # 屏蔽基波

    # 修正: 如果启用了DC屏蔽，SFDR计算也应该屏蔽DC
    if dc_mask_width >= 0:
        dc_idx = N // 2
        psd_masked[max(0, dc_idx - dc_mask_width):min(N, dc_idx + dc_mask_width + 1)] = -200

    # 修正: 如果启用了镜像屏蔽，SFDR计算也应该屏蔽镜像
    if exclude_image:
        dc_idx = N // 2
        fund_offset = peak_idx - dc_idx
        image_idx = dc_idx - fund_offset
        if 0 <= image_idx < N:
            psd_masked[max(0, image_idx - image_span):min(N, image_idx + image_span + 1)] = -200

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
    channel_energy_lin = np.sum(psd_energy[center - q_span:center + q_span])

    # 如果启用了 DC 屏蔽 (>=0)，Channel Power 也应该扣除 DC 能量
    if dc_mask_width >= 0:
        dc_idx = N // 2
        # 确保扣除范围在 Channel 带宽内
        mask_start = max(center - q_span, dc_idx - dc_mask_width)
        mask_end = min(center + q_span, dc_idx + dc_mask_width + 1)
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

    return (
        fund_freq,
        fund_power,
        total_power,
        channel_power,
        snr,
        sfdr,
        noise_per_hz,
    )
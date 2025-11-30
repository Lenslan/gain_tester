use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use num_complex::Complex64;
use rust_xlsxwriter::{ColNum, Color, Format, FormatAlign, RowNum, Workbook, Worksheet};
use rustfft::FftPlanner;
use crate::config::Band;

#[derive(Debug)]
struct RfMetrics {
    fund_freq: f64,
    fund_power: f64,
    total_power: f64,
    channel_power: f64,
    snr: f64,
    sfdr: f64,
    noise_per_hz: f64
}

impl RfMetrics {
    fn new(fund_freq: f64, fund_power: f64, total_power: f64, channel_power: f64, snr: f64, sfdr: f64, noise_per_hz: f64) -> Self {
        Self {
            fund_freq,
            fund_power,
            total_power,
            channel_power,
            snr,
            sfdr,
            noise_per_hz,
        }
    }
}

trait CalcMetric {
    fn get_iq_data(&self) -> (Vec<i16>, Vec<i16>, u8);
    fn calc_metric(&self) -> RfMetrics {
        let (i_data, q_data, fs) = self.get_iq_data();
        // let code = std::fs::read_to_string("python/calc_rf_metrics.py")?;
        // Python::with_gil(|py| {
        //     let module = PyModule::from_code(py,
        //                                      &CString::new(code).unwrap(),
        //                                      &CString::new("calc_rf_metrics.py").unwrap(),
        //                                      &CString::new("calc_rf_metrics").unwrap())?;
        //     let res = module
        //         .getattr("get_rf_metrics")?
        //         .call1((i_data, q_data, fs))?
        //         .extract()?;
        //     let (fund_freq, fund_power, total_power, channel_power, snr, sfdr, noise_per_hz) = res;
        //     Ok(RfMetrics::new(fund_freq, fund_power, total_power, channel_power, snr, sfdr, noise_per_hz))
        // })

        // === 配置参数 ===
        let power_offset_db = -0.004;
        let dc_mask_width = 2_isize; // 使用 isize 以方便计算索引差
        let fund_span = 10_isize;
        let exclude_image = true;
        let image_span = 1_isize;
        let noise_hann_correction = true;
        let norm_factor = 2047.0;

        let fs = fs as f64 * 1e6;
        let n = i_data.len();
        assert_eq!(n, q_data.len(), "I and Q data length must match");

        // === 1. 数据准备与归一化 ===
        let mut complex_data: Vec<Complex64> = i_data
            .iter()
            .zip(q_data.iter())
            .map(|(&i, &q)| {
                Complex64::new(
                    i as f64 / norm_factor,
                    q as f64 / norm_factor
                )
            })
            .collect();

        // === 2. 加窗 (Blackman) ===
        let mut window = Vec::with_capacity(n);
        let mut s2_acc = 0.0; // sum(window^2)
        let mut cg_acc = 0.0; // sum(window)

        for i in 0..n {
            let val = 0.42
                - 0.5 * (2.0 * PI * i as f64 / n as f64).cos()
                + 0.08 * (4.0 * PI * i as f64 / n as f64).cos();
            window.push(val);

            // 应用窗口
            complex_data[i] = complex_data[i] * val;

            s2_acc += val * val;
            cg_acc += val;
        }

        // === 3. FFT ===
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);
        fft.process(&mut complex_data);

        // FFTShift: 将零频移到中心
        // 对应 np.fft.fftshift (对于偶数长度，左旋 N/2)
        let shift_amount = n / 2;
        complex_data.rotate_left(shift_amount);

        // 频率轴计算 (shifted)
        // Python: freqs = fftshift(fftfreq(N, 1.0))
        // fftfreq 生成 [0, 1, ..., n/2-1, -n/2, ..., -1] / n
        // shift 后: [-n/2, ..., -1, 0, 1, ..., n/2-1] / n
        let freqs_normalized: Vec<f64> = (0..n)
            .map(|i| (i as f64 - (n as f64 / 2.0)) / n as f64)
            .collect();

        // === 4. 功率谱计算 (校准) ===
        let s2 = s2_acc / n as f64;
        let cg = cg_acc / n as f64;

        // ENBW
        let enbw = s2 / (cg * cg);

        // 计算 Magnitude Spectrum (Display) 和 Energy PSD (Statistics)
        let mut psd_display = Vec::with_capacity(n);
        let mut psd_energy = Vec::with_capacity(n);

        for val in &complex_data {
            let abs_val = val.norm(); // equivalent to np.abs

            // Display: 20 * log10(abs / (N * CG))
            let mag_spec = abs_val / (n as f64 * cg);
            psd_display.push(20.0 * (mag_spec + 1e-12).log10());

            // Energy: (abs / N)^2 / S2
            let energy_val = (abs_val / n as f64).powi(2) / s2;
            psd_energy.push(energy_val);
        }

        // === 5. 信号参数计算 ===

        // 1. Total Power
        let total_power_lin: f64 = psd_energy.iter().sum();
        let total_power = 10.0 * (total_power_lin + 1e-12).log10() + power_offset_db;

        // 2. Fund Power (Find Peak)
        let dc_idx = n / 2; // Center index
        let peak_idx;

        if dc_mask_width >= 0 {
            // 寻找除了 DC 范围之外的最大值
            let mask_start = (dc_idx as isize - dc_mask_width).max(0) as usize;
            let mask_end = (dc_idx as isize + dc_mask_width + 1).min(n as isize) as usize;

            peak_idx = psd_energy.iter().enumerate()
                .filter(|(i, _)| *i < mask_start || *i >= mask_end)
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0); // fallback
        } else {
            peak_idx = psd_energy.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
        }

        // 积分基波能量
        let fund_start = (peak_idx as isize - fund_span).max(0) as usize;
        let fund_end = (peak_idx as isize + fund_span + 1).min(n as isize) as usize;

        let fund_energy_lin: f64 = psd_energy[fund_start..fund_end].iter().sum();
        let fund_power = 10.0 * (fund_energy_lin + 1e-12).log10() + power_offset_db;

        // 基波频率
        let fund_freq = freqs_normalized[peak_idx] * fs / 1e6; // MHz

        // 3. Noise Power & SNR
        let mut noise_power_lin = total_power_lin - fund_energy_lin;

        // 减去 DC 能量
        if dc_mask_width >= 0 {
            let dc_start = (dc_idx as isize - dc_mask_width).max(0) as usize;
            let dc_end = (dc_idx as isize + dc_mask_width + 1).min(n as isize) as usize;
            let dc_energy_masked: f64 = psd_energy[dc_start..dc_end].iter().sum();
            noise_power_lin -= dc_energy_masked;
        }

        // 减去镜像能量
        if exclude_image {
            let fund_offset = peak_idx as isize - dc_idx as isize;
            let image_idx = dc_idx as isize - fund_offset;

            if image_idx >= 0 && image_idx < n as isize {
                let img_idx_u = image_idx as usize;
                let img_start = (img_idx_u as isize - image_span).max(0) as usize;
                let img_end = (img_idx_u as isize + image_span + 1).min(n as isize) as usize;
                let image_energy: f64 = psd_energy[img_start..img_end].iter().sum();

                noise_power_lin -= image_energy;
            }
        }

        if noise_power_lin <= 1e-15 {
            noise_power_lin = 1e-15;
        }

        let snr = 10.0 * (fund_energy_lin / noise_power_lin).log10();

        // 4. SNRFS (Unused in return but calculated in Python)
        let noise_power_db = 10.0 * noise_power_lin.log10();
        // let snrfs = 0.0 - noise_power_db;

        // 5. SFDR
        // 需要创建一个 masked 的 psd_display 来寻找最大杂散
        // 注意：Rust 这里我们不做真正的数组拷贝修改，而是在寻找最大值时进行过滤
        let mask_span = 6_isize;

        // 定义一个闭包来判断是否被 Mask
        let is_masked = |idx: usize| -> bool {
            let i_isize = idx as isize;
            // Mask Fundamental
            if i_isize >= peak_idx as isize - mask_span && i_isize < peak_idx as isize + mask_span + 1 {
                return true;
            }
            // Mask DC
            if dc_mask_width >= 0 {
                if i_isize >= dc_idx as isize - dc_mask_width && i_isize < dc_idx as isize + dc_mask_width + 1 {
                    return true;
                }
            }
            // Mask Image
            if exclude_image {
                let fund_offset = peak_idx as isize - dc_idx as isize;
                let image_idx = dc_idx as isize - fund_offset;
                if i_isize >= image_idx - image_span && i_isize < image_idx + image_span + 1 {
                    return true;
                }
            }
            false
        };

        let spur_peak = psd_display.iter().enumerate()
            .filter(|(i, _)| !is_masked(*i))
            .map(|(_, &val)| val)
            .fold(f64::NEG_INFINITY, f64::max); // 找最大值，初始值为负无穷

        // 如果全部都被mask了（极端情况），给予一个默认极小值
        let spur_peak = if spur_peak == f64::NEG_INFINITY { -200.0 } else { spur_peak };

        let sfdr = spur_peak - fund_power;

        // 6. Noise/Hz
        let enbw_val = if noise_hann_correction { 1.5 } else { enbw };
        let noise_per_hz = noise_power_db - 10.0 * fs.log10() - 10.0 * enbw_val.log10();

        // 7. Channel Power
        let q_span = n / 4;
        let ch_start = (dc_idx as isize - q_span as isize).max(0) as usize;
        let ch_end = (dc_idx as isize + q_span as isize).min(n as isize) as usize;

        let mut channel_energy_lin: f64 = psd_energy[ch_start..ch_end].iter().sum();

        if dc_mask_width >= 0 {
            let mask_start = (ch_start as isize).max(dc_idx as isize - dc_mask_width) as usize;
            let mask_end = (ch_end as isize).min(dc_idx as isize + dc_mask_width + 1) as usize;

            if mask_end > mask_start {
                let dc_energy_in_channel: f64 = psd_energy[mask_start..mask_end].iter().sum();
                channel_energy_lin -= dc_energy_in_channel;
                if channel_energy_lin < 1e-15 { channel_energy_lin = 1e-15; }
            }
        }

        let channel_power = 10.0 * (channel_energy_lin + 1e-12).log10() + power_offset_db;

        // 8. DC Power (Unused in return)
        // let dc_power_lin = psd_energy[dc_idx];
        // let dc_power = 10.0 * (dc_power_lin + 1e-12).log10() + power_offset_db;

        // 9. Average Bin Noise (Unused in return)
        // let avg_bin_noise_lin = noise_power_lin / n as f64;
        // let avg_bin_noise = 10.0 * (avg_bin_noise_lin + 1e-12).log10();

        RfMetrics::new (
            fund_freq,
            fund_power,
            total_power,
            channel_power,
            snr,
            sfdr,
            noise_per_hz,
        )
    }
}

impl CalcMetric for (Vec<i16>, Vec<i16>, u8) {
    fn get_iq_data(&self) -> (Vec<i16>, Vec<i16>, u8) {
        (self.0.clone(), self.1.clone(), self.2)
    }
}

pub(crate) struct FileParser {
    file_list: Vec<String>,
    workbook: Workbook
}

impl FileParser {
    pub(crate) fn new(file_list: Vec<String>) -> Self {
        let workbook = Workbook::new();
        Self {
            file_list,
            workbook
        }
    }

    pub fn add_file(&mut self, filename: String) {
        self.file_list.push(filename);
    }

    pub fn sort_file(mut self) -> Self {
        self.file_list.sort();
        self
    }

    pub fn parse_and_write(&mut self) -> anyhow::Result<()>{
        self.write_band_excel(Band::HB)?;
        self.write_band_excel(Band::LB)?;

        self.workbook.save("iq_dump/result.xlsx")?;
        Ok(())
    }

    fn write_band_excel(&mut self, band: Band) -> anyhow::Result<()> {
        let mut line = 2;
        let sheet = self.workbook.add_worksheet();

        Self::write_header(sheet)?;

        let band_name = format!("{}", band).to_lowercase();
        self.file_list.iter()
            .for_each(|f| {
                let file = Path::new(f)
                    .file_name()
                    .and_then(|x| x.to_str())
                    .unwrap();
                if file.starts_with(&band_name) {
                    // hb_iq_{fem}_{lna}_{vga}.txt
                    let res = Self::parse_file(f, 40);
                    // Some((f[6..12].into_string(), res))
                    Self::write_excel(sheet, line, res, &file[6..12]).unwrap();
                    line += 1;
                }
            });
        log::info!("{} has {} cases", band, line-2);
        sheet.set_name(format!("{}", band))?;
        Ok(())
    }

    fn write_header(sheet: &mut Worksheet) -> anyhow::Result<()> {
        let header_format = Format::new()
            .set_bold()
            .set_align(FormatAlign::Center)
            .set_align(FormatAlign::VerticalCenter)
            .set_background_color(Color::Gray);
        let path_format = Format::new()
            .set_bold()
            .set_align(FormatAlign::VerticalCenter)
            .set_align(FormatAlign::Center);
        sheet.set_row_height(0, 32)?;
        sheet.merge_range(0, 1, 0, 4, "Path1", &path_format)?;
        sheet.merge_range(0, 6, 0, 9, "Path2", &path_format)?;

        sheet.set_row_height(1, 28)?;
        let header = ["Gain", "Fund_freq", "Fund_power", "Total_power", "Channel_power"];
        for (idx, item) in header.iter().enumerate() {
            if idx == 0 {
                sheet.set_column_width(idx as ColNum, 32)?;
            } else {
                sheet.set_column_width(idx as ColNum, 22)?;
            }

            sheet.write_with_format(1, idx as ColNum, *item, &header_format)?;
        }
        for (idx, item) in header[1..].iter().enumerate() {
            sheet.set_column_width(idx as ColNum + 6, 22)?;
            sheet.write_with_format(1, idx as ColNum + 6, *item, &header_format)?;
        }

        Ok(())

    }

    fn write_excel(sheet: &mut Worksheet, line: RowNum, metrics: (RfMetrics, RfMetrics), gain: &str) -> anyhow::Result<()> {
        sheet.write(line, 0, gain)?;
        sheet.write(line, 1, metrics.0.fund_freq)?;
        sheet.write(line, 2, metrics.0.fund_power)?;
        sheet.write(line, 3, metrics.0.total_power)?;
        sheet.write(line, 4, metrics.0.channel_power)?;
        sheet.write(line, 6, metrics.1.fund_freq)?;
        sheet.write(line, 7, metrics.1.fund_power)?;
        sheet.write(line, 8, metrics.1.total_power)?;
        sheet.write(line, 9, metrics.1.channel_power)?;
        Ok(())

    }

    fn parse_file(filename: &str, fs: u8) -> (RfMetrics, RfMetrics) {
        let file = File::open(filename).unwrap();
        let mut i_data_path1 = Vec::new();
        let mut q_data_path1 = Vec::new();
        let mut i_data_path2 = Vec::new();
        let mut q_data_path2 = Vec::new();
        let mut temp_flag = true;
        for line in BufReader::new(file).lines() {
            let line = &line.unwrap();
            if temp_flag {
                if line.is_empty() { continue; }
                if line.starts_with("0x00") {
                    temp_flag = false;
                    i_data_path1.push(hex12_to_i16(u16::from_str_radix(&line[7..10], 16).unwrap()));
                    q_data_path1.push(hex12_to_i16(u16::from_str_radix(&line[4..7], 16).unwrap()));
                    continue
                }
            } else {
                if line.is_empty() { continue; }
                if line.starts_with("0x00") {
                    temp_flag = true;
                    i_data_path2.push(hex12_to_i16(u16::from_str_radix(&line[7..10], 16).unwrap()));
                    q_data_path2.push(hex12_to_i16(u16::from_str_radix(&line[4..7], 16).unwrap()));
                    continue
                }
            }
        }
        if cfg!(test) {
            println!("{:?}", i_data_path1);
            println!("{:?}", q_data_path1);
        }
        let res1 = (i_data_path1, q_data_path1, fs).calc_metric();
        let res2 = (i_data_path2, q_data_path2, fs).calc_metric();
        (res1, res2)
    }
}

fn hex12_to_i16(value: u16) -> i16 {
    let raw = value & 0x0fff;
    if raw & 0x0800 != 0 {
        (raw | 0xf000) as i16
    } else {
        raw as i16
    }
}

#[cfg(test)]
mod tests {
    use crate::rfmetrics::FileParser;

    // #[test]
    // fn test_calc_metric() {
    //     let file = String::from("test/iq-success.txt");
    //     let res = FileParser::parse_file(&file, 40);
    //     println!("{:?}", res);
    // }

    #[test]
    fn test_excel() {
        simple_logger::init_with_level(log::Level::Info).unwrap();
        // let file_list = vec!["test/hb_iq_0_0_00.txt"];
        let mut file = FileParser::new(Vec::new());

        file.add_file("test/hb_iq_0_0_00.txt".into());
        file.sort_file()
            .parse_and_write()
            .unwrap();
    }
}
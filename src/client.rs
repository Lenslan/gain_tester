use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::net::TcpStream;
use std::path::Path;
use anyhow::{anyhow, Context};
use serde::{Deserialize, Serialize};
use crate::config::{Band, TestBand};
use crate::rfmetrics::FileParser;
use crate::testcase::TestCase;

#[derive(Serialize, Deserialize, Debug)]
enum DumpCommand {
    DumpIQ{
        band_5g: bool,
        file_name: String
    },
    DelFiles,
    CopyFiles(String),
    SetReg{
        addr: u32,
        value: u32
    },
    ShellCmd(String),
    ATEInit,
    ATECmd{
        cmd: String,
        args: Vec<String>
    },
}

#[derive(Serialize, Deserialize, Debug)]
struct  ResponseHeader {
    is_error: bool,
    file_size: u64,
}

pub struct Dut {
    stream: TcpStream,
    reader: BufReader<TcpStream>,
    pub(crate) file_list: FileParser
}

impl Dut {
    pub fn new(addr: &str) -> Dut {
        let stream = TcpStream::connect(addr).expect("Could not connect to server");
        let reader = BufReader::new(stream.try_clone().expect("Could not clone stream"));
        Dut {
            stream,
            reader,
            file_list: FileParser::new(Vec::new())
        }
    }

    fn handle_resp(&mut self) -> anyhow::Result<ResponseHeader> {
        let mut header_line = String::new();
        self.reader.read_line(&mut header_line)?;
        let resp: ResponseHeader = serde_json::from_str(&header_line)
            .with_context(|| format!("Could not parse response header line {}", header_line))?;
        Ok(resp)
    }

    fn send_cmd(&mut self, cmd: DumpCommand) -> anyhow::Result<()> {
        let json_req = serde_json::to_string(&cmd)?;
        self.stream.write_all(json_req.as_bytes())?;
        self.stream.write_all(b"\n")?;
        Ok(())
    }

    pub fn dump_iq(&mut self, band_5g: Band, file_name: String) -> anyhow::Result<bool> {
        // Send command
        let cmd = if band_5g == Band::HB {
            "echo 0 1 0 15 0 e000 0 2 0  1 0 0 0 > /sys/kernel/debug/ieee80211/phy2/siwifi/iq_engine"
        } else {
            "echo 0 1 0 15 0 1c000 0 2 0  1 0 0 0 > /sys/kernel/debug/ieee80211/phy0/siwifi/iq_engine"
        };
        let cmd = DumpCommand::ShellCmd(cmd.into());
        self.send_cmd(cmd)?;
        let cmd = DumpCommand::DumpIQ{band_5g: band_5g == Band::HB, file_name};
        self.send_cmd(cmd)?;

        // read response
        Ok(!self.handle_resp()?.is_error)
    }

    pub fn del_files(&mut self) -> anyhow::Result<bool> {
        //send command
        let cmd = DumpCommand::DelFiles;
        self.send_cmd(cmd)?;

        //read response
        Ok(!self.handle_resp()?.is_error)
    }

    pub fn copy_files(&mut self, file_name: String) -> anyhow::Result<bool> {
        let cmd = DumpCommand::CopyFiles(file_name.clone());
        self.send_cmd(cmd)?;
        if !Path::new("./iq_dump").exists() {
            fs::create_dir_all("./iq_dump")?;
        }
        //read response
        let res = self.handle_resp()?;
        if res.is_error {
            log::error!("Could not copy files! {}", file_name);
            Err(anyhow!("Could not copy files!"))
        } else {
            log::info!("Copy file ing...");
            let mut buffer = vec![0u8; 64*1024];
            let mut remaining = res.file_size;
            let mut file = BufWriter::new(File::create(format!("./iq_dump/{}", file_name))?);  //TODO;

            while remaining > 0 {
                let read_len = std::cmp::min(remaining, buffer.len() as u64) as usize;
                let n = self.reader.read(&mut buffer[..read_len])?;
                if n == 0 {
                    return Err(anyhow!("Not completely receive file!"));
                }
                file.write_all(&buffer[..n])?;
                remaining -= n as u64;
            }

            file.flush()?;
            log::info!("Saved file {}", file_name);
            self.file_list.add_file(format!("./iq_dump/{}", file_name));
            Ok(true)
        }
    }

    pub fn fix_gain(&mut self, is_hb:Band, fem: u8, lna: u8, vga: u8) -> anyhow::Result<()> {
        // devmem 0x30c02f88 32 0x2d170d17
        // devmem 0x30c02f88 32 0x3d171d17
        // devmem 0x30c02f88 32 0x24000400
        // devmem 0x30c02f88 32 0x34001400
        //
        // devmem 0x20c02f88 32 0x2d170d17
        // devmem 0x20c02f88 32 0x3d171d17
        // devmem 0x20c02f88 32 0x24000400
        // devmem 0x20c02f88 32 0x34001400
        let addr = if is_hb == Band::HB {
            0x30c02f88
        } else {
            0x20c02f88
        };
        let gain_value = pack_bit(fem, lna, vga);

        let cmd = DumpCommand::SetReg {addr, value: 0x2d170d17};
        self.send_cmd(cmd)?;

        let cmd = DumpCommand::SetReg {addr, value: 0x3d171d17};
        self.send_cmd(cmd)?;

        let value = (gain_value as u32 | 0x2000) << 16 | (gain_value as u32 | 0x0000);
        let cmd = DumpCommand::SetReg {addr, value};
        self.send_cmd(cmd)?;

        let value = (gain_value as u32 | 0x3000) << 16 | (gain_value as u32 | 0x1000);
        let cmd = DumpCommand::SetReg {addr, value};
        self.send_cmd(cmd)?;

        Ok(())
    }

    pub fn shut_down_band(&mut self, band_5g: Band) -> anyhow::Result<()> {
        // echo 20000000.wmac > /sys/bus/platform/drivers/siwifi_umac/unbind
        // devmem 0x04e00030 32 0xffff
        // devmem 0x04e00478 32 7
        // devmem 0x04e004c8 32 7
        let cmd = if band_5g == Band::HB {
            "echo 30000000.wmac > /sys/bus/platform/drivers/siwifi_umac/unbind"
        } else {
            "echo 20000000.wmac > /sys/bus/platform/drivers/siwifi_umac/unbind"
        };
        let cmd = DumpCommand::ShellCmd(cmd.into());
        self.send_cmd(cmd)?;

        let cmd = DumpCommand::SetReg { addr: 0x04e00030, value: 0xffff};
        self.send_cmd(cmd)?;

        let cmd = DumpCommand::SetReg { addr: 0x04e00478, value: 7};
        self.send_cmd(cmd)?;

        let cmd = DumpCommand::SetReg { addr: 0x04e004c8, value: 7};
        self.send_cmd(cmd)?;

        log::info!("Shut {} Donw Over!", band_5g);
        Ok(())
    }

    pub fn shut_up_band(&mut self, band_5g: Band) -> anyhow::Result<()> {
        let cmd = if band_5g == Band::HB {
            "echo 30000000.wmac > /sys/bus/platform/drivers/siwifi_umac/bind"
        } else {
            "echo 20000000.wmac > /sys/bus/platform/drivers/siwifi_umac/bind"
        };
        let cmd = DumpCommand::ShellCmd(cmd.into());
        self.send_cmd(cmd)?;

        let args = if band_5g == Band::HB {
            ["wlan0", "up"]
        } else {
            ["wlan1", "up"]
        }
            .iter()
            .map(|s| {s.to_string()})
            .collect::<Vec<_>>();
        let cmd = DumpCommand::ATECmd {cmd: "ifconfig".into(), args};
        self.send_cmd(cmd)
    }

    pub fn ate_init(&mut self) -> anyhow::Result<()> {
        self.send_cmd(DumpCommand::ATEInit)?;
        let res = self.handle_resp()?;

        log::info!("Ate init status error?{}", res.is_error);
        Ok(())
    }

    pub fn open_rx(&mut self, is_hb: Band) -> anyhow::Result<()> {
        let args = if is_hb == Band::HB {
            "wlan0 fastconfig -f 5180 -c 5180 -w 1 -u 1 -r"
        } else {
            "wlan1 fastconfig -f 2412 -c 2412 -w 1 -u 1 -r"
        }
            .trim()
            .split(" ")
            .map(|s| s.to_string())
            .collect::<Vec<String>>();
        let cmd = DumpCommand::ATECmd{cmd: "ate_cmd".into(), args};
        self.send_cmd(cmd)
    }

    pub fn close_rx(&mut self, is_hb: Band) -> anyhow::Result<()> {
        let args = if is_hb == Band::HB {
            "wlan0 fastconfig -k"
        } else {
            "wlan1 fastconfig -k"
        }
            .trim()
            .split(" ")
            .map(|s| s.to_string())
            .collect::<Vec<String>>();
        let cmd = DumpCommand::ATECmd{cmd: "ate_cmd".into(), args};
        self.send_cmd(cmd)
    }

    pub fn run_test(&mut self, band: TestBand) {
        band.run_test(self)
    }
}

fn pack_bit(a: u8, b: u8, c:u8) -> u16 {
    let bit1 = a & 0b0000_0001;
    let bit2 = b & 0b0000_0111;
    let bit3 = c & 0b0001_1111;

    ((bit1 as u16) << 8 | (bit2 as u16) << 5 | (bit3 as u16)) << 1 | (1 << 10)
}


#[cfg(test)]
mod test {
    use crate::client::pack_bit;

    #[test]
    fn tset_pack_bit() {
        println!("0x{:08X}", pack_bit(1, 0, 1));
    }
}
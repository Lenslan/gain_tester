use std::fs::File;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::Path;
use std::thread::sleep;
use std::time::Duration;
use serialport::{SerialPort};
use anyhow::{Context, Result};
use ssh2::Session;

pub struct Dut {
    inst: Box<dyn SerialPort>,
}

impl Dut {
    pub fn new(port_name: &'static str) -> Self {
        let builder = serialport::new(port_name, 115200)
            .timeout(Duration::from_millis(1000));
        let inst = builder.open().unwrap_or_else(|e| {
            eprintln!("Connect Serial Error: {}\n Close IPOP??", e);
            std::process::exit(1);
        });
        Self {
            inst
        }
    }

    pub fn write_cmd(&mut self, cmd: &str) -> Result<()> {
        let cmd = format!("{}\r\n", cmd);
        self.inst.write(&cmd.as_bytes())
            .with_context(|| format!("Failed to write to serial port: {:?}", cmd))?;
        self.inst.flush()
            .with_context(|| format!("Failed to flush serial port: {:?}", cmd))?;
        sleep(Duration::from_millis(100));
        Ok(())
    }

    pub fn dump_lb_iq(&mut self, iq_file: &str) -> Result<()> {
        self.write_cmd("echo 0 1 0 15 0 1c000 0 2 0  1 0 0 0 > /sys/kernel/debug/ieee80211/phy0/siwifi/iq_engine")?;
        let cmd = format!("{}{}", "memdump 0x30000000 0xd8000 | hexdump  -v -e \'\"0x%08x\"\"\n\"\' > /tmp/", iq_file);
        self.write_cmd(&cmd)?;
        log::info!("dump_lb_iq Over");
        Ok(())
    }

    pub fn dump_hb_iq(&mut self, iq_file: &str) -> Result<()> {
        self.write_cmd("echo 0 1 0 15 0 e000 0 2 0  1 0 0 0 > /sys/kernel/debug/ieee80211/phy1/siwifi/iq_engine")?;
        let cmd = format!("{}{}", "memdump 0x20000000 0x62000 | hexdump  -v -e \'\"0x%08x\"\"\n\"\' > /tmp/", iq_file);
        self.write_cmd(&cmd)?;
        log::info!("dump_hb_iq Over");
        Ok(())
    }

    pub fn delete_files(&mut self) -> Result<()> {
        self.write_cmd("rm -rf /tmp/*.txt")?;
        log::info!("delete_files OK");
        Ok(())
    }

}

pub fn copy_file_from_dut(file_list: &[String]) -> Result<()> {
    let tcp = TcpStream::connect(("192.168.1.1:22"))?;
    let mut sess = Session::new()?;
    sess.set_tcp_stream(tcp);
    sess.handshake()?;

    // sess.userauth_password("root", "")?;
    sess.userauth_pubkey_file(
        "root",
        None,
        &Path::join(&dirs::home_dir().unwrap(), ".ssh/my_isa"),
        None
    )?;

    let sftp = sess.sftp()?;
    for file in file_list {
        let file_name_remote = format!("/tmp/{}", file);
        let file_name_local = format!("./iq_dump/{}", file);
        let mut remote = sftp.open(&file_name_remote)?;
        let mut local = File::create(file_name_local)?;

        let mut buffer = [0u8; 4096];
        loop {
            let n = remote.read(&mut buffer)?;
            if n == 0 {
                log::info!("Dump file {} Over", file);
                break;
            }
            local.write_all(&buffer[..n])?;
        }
    }

    Ok(())
}

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::net::TcpStream;
use anyhow::{anyhow, Context};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
enum DumpCommand {
    DumpIQ{
        band_5g: bool,
        file_name: String
    },
    DelFiles,
    CopyFiles(String)
}

#[derive(Serialize, Deserialize, Debug)]
struct  ResponseHeader {
    is_error: bool,
    file_size: u64,
}

pub struct Dut {
    stream: TcpStream,
    reader: BufReader<TcpStream>,
}

impl Dut {
    pub fn new(addr: &str) -> Dut {
        let stream = TcpStream::connect(addr).expect("Could not connect to server");
        let reader = BufReader::new(stream.try_clone().expect("Could not clone stream"));
        Dut {
            stream,
            reader
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

    pub fn dump_iq(&mut self, band_5g: bool, file_name: String) -> anyhow::Result<bool> {
        // Send command
        let cmd = DumpCommand::DumpIQ{band_5g, file_name};
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

        //read response
        let res = self.handle_resp()?;
        if res.is_error {
            log::error!("Could not copy files! {}", file_name);
            Err(anyhow!("Could not copy files!"))
        } else {
            log::info!("Copy file ing...");
            let mut buffer = vec![0u8; 64*1024];
            let mut remaining = res.file_size;
            let mut file = BufWriter::new(File::create(format!("./iq_dump/{}", file_name))?);

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
            Ok(true)
        }
    }
}
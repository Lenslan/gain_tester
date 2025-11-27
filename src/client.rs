use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
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
}

impl Dut {
    pub fn new(addr: &str) -> Dut {
        let stream = TcpStream::connect(addr).expect("Could not connect to server");
        Dut {
            stream
        }
    }

    fn handle_resp(&self) -> anyhow::Result<ResponseHeader> {
        let mut reader = BufReader::new(&self.stream);
        let mut header_line = String::new();
        reader.read_line(&mut header_line)?;
        let resp: ResponseHeader = serde_json::from_str(&header_line)
            .with_context(|| format!("Could not parse response header line {}", header_line))?;
        Ok(resp)
    }

    pub fn dump_iq(&mut self, band_5g: bool, file_name: String) -> anyhow::Result<bool> {
        // Send command
        let cmd = DumpCommand::DumpIQ{band_5g, file_name};
        let json_req = serde_json::to_string(&cmd)?;
        self.stream.write_all(json_req.as_bytes())?;
        self.stream.write_all(b"\n")?;

        // read response
        Ok(!self.handle_resp()?.is_error)
    }

    pub fn del_files(&mut self) -> anyhow::Result<bool> {
        //send command
        let cmd = DumpCommand::DelFiles;
        let json_req = serde_json::to_string(&cmd)?;
        self.stream.write_all(json_req.as_bytes())?;
        self.stream.write_all(b"\n")?;

        //read response
        Ok(!self.handle_resp()?.is_error)
    }

    pub fn copy_files(&mut self, file_name: String) -> anyhow::Result<bool> {
        let cmd = DumpCommand::CopyFiles(file_name.clone());
        let json_req = serde_json::to_string(&cmd)?;
        self.stream.write_all(json_req.as_bytes())?;
        self.stream.write_all(b"\n")?;

        //read response
        let res = self.handle_resp()?;
        if res.is_error {
            log::error!("Could not copy files!");
            Err(anyhow!("Could not copy files!"))
        } else {
            let mut buffer = vec![0; res.file_size as usize];
            let mut reader = BufReader::new(&self.stream);
            reader.read_exact(&mut buffer)
                .context("Read file fail")?;

            let mut file = File::create(format!("./iq_dump/{}", file_name))?;
            file.write_all(&buffer)?;
            log::info!("Saved file {}", file_name);
            Ok(true)
        }
    }
}
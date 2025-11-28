use std::io;
use crate::client::Dut;
use crate::config::Band::{HB, LB};

mod client;
mod config;

fn main() -> anyhow::Result<()> {
    simple_logger::init_with_level(log::Level::Info)?;

    log::warn!("Remember to open IQ instrument!!!");

    let mut dut = Dut::new("192.168.1.1:9600");

    let mut temp = String::new();
    dut.ate_init()?;
    // io::stdin().read_line(&mut temp)?;
    dut.shut_down_band(HB)?;
    // io::stdin().read_line(&mut temp)?;
    dut.open_rx(LB)?;


    for i in 0..3 {
        dut.fix_gain(LB, 0,0,i.into())?;
        let iq_name = format!("lb_test_iq{}.txt", i);
        dut.dump_iq(LB, iq_name.clone())?;
        dut.copy_files(iq_name)?;
        let status = dut.del_files()?;
        println!("delete status {}", status);
    }

    dut.close_rx(LB)?;
    dut.shut_up_band(HB)?;
    dut.shut_down_band(LB)?;
    dut.open_rx(HB)?;

    for i in 0..3 {
        dut.fix_gain(HB, 0,0,i.into())?;
        let iq_name = format!("hb_test_iq{}.txt", i);
        dut.dump_iq(HB, iq_name.clone())?;
        dut.copy_files(iq_name)?;
        let status = dut.del_files()?;
        println!("delete status {}", status);
    }

    println!("TestDone");
    Ok(())
}

use crate::client::Dut;
use crate::config::Band::{HB, LB};
use crate::config::GainType::{Fem, Lna, Vga};
use crate::config::TestBand;
use crate::testcase::TestCase;

mod client;
mod config;
mod rfmetrics;
mod testcase;

fn main() -> anyhow::Result<()> {
    simple_logger::init_with_level(log::Level::Info)?;

    log::warn!("Remember to open IQ instrument!!!");

    let mut dut = Dut::new("192.168.1.1:9600");

    dut.ate_init()?;
    dut.shut_down_band(HB)?;
    dut.open_rx(LB)?;

    dut.run_test(TestBand::LB(Fem(0..2)));
    dut.run_test(TestBand::LB(Lna(0..8)));
    dut.run_test(TestBand::LB(Vga(0..21)));

    dut.close_rx(LB)?;
    dut.shut_up_band(HB)?;
    dut.shut_down_band(LB)?;
    dut.open_rx(HB)?;

    dut.run_test(TestBand::HB(Fem(0..2)));
    dut.run_test(TestBand::HB(Lna(0..8)));
    dut.run_test(TestBand::HB(Vga(0..21)));

    dut.file_list
        .sort_file()
        .parse_and_write()?;

    println!("TestDone");
    Ok(())
}

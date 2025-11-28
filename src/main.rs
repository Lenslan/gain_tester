use crate::client::Dut;

mod client;

fn main() -> anyhow::Result<()> {
    simple_logger::init_with_level(log::Level::Info)?;

    log::warn!("Remember to open IQ instrument!!!");

    let mut dut = Dut::new("192.168.1.1:9600");
    // dut.dump_iq(true, "hb_test_iq.txt".into())?;
    dut.copy_files("hb_test_iq.txt".into())?;
    dut.del_files()?;
    println!("TestDone");
    Ok(())
}

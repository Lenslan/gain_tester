use crate::serial::{copy_file_from_dut, Dut};

mod serial;

fn main() -> anyhow::Result<()> {
    simple_logger::init_with_level(log::Level::Info)?;

    log::warn!("Remember to open IQ instrument!!!");

    let mut dut = Dut::new("COM7");

    dut.dump_hb_iq("test_iq1.txt")?;
    dut.dump_hb_iq("test_iq2.txt")?;
    dut.dump_hb_iq("test_iq3.txt")?;

    let file_list = vec![
        String::from("test_iq1.txt"),
        String::from("test_iq2.txt"),
        String::from("test_iq3.txt"),
    ];

    copy_file_from_dut(&file_list)?;

    Ok(())
}

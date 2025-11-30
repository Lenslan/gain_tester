use std::ops::Range;
use strum::Display;

#[derive(PartialEq, Eq, Debug, Display, Clone, Copy)]
pub enum Band {
    HB,
    LB
}

pub enum TestBand {
    HB(GainType),
    LB(GainType)
}

impl TestBand {
    pub fn return_gain_type(&self) -> GainType {
        match self {
            TestBand::HB(item) => {item.return_type()}
            TestBand::LB(item) => {item.return_type()}
        }
    }
}

pub enum GainType {
    Fem(Range<u8>),
    Lna(Range<u8>),
    Vga(Range<u8>)
}

impl GainType {
    pub fn return_iter(&self) -> Range<u8> {
        match self {
            GainType::Fem(item) => {item.clone()}
            GainType::Lna(item) => {item.clone()}
            GainType::Vga(item) => {item.clone()}
        }
    }
    
    fn return_type(&self) -> Self {
        use GainType::*;
        match self {
            Fem(_) => {Fem(0..0)}
            Lna(_) => {Lna(0..0)}
            Vga(_) => {Vga(0..0)}
        }
    }
}

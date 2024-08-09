#[cfg(target_pointer_width = "64")]
pub type HalfSizeNative = u32;
#[cfg(target_pointer_width = "32")]
pub type HalfSizeNative = u16;
#[cfg(target_pointer_width = "16")]
pub type HalfSizeNative = u8;
pub const HALF_SIZE_BYTES: usize = HalfSizeNative::BITS as usize / 8;
pub const FULL_SIZE_BYTES: usize = usize::BITS as usize / 8;
const _: () = {
    #[allow(clippy::manual_assert)]
    if HALF_SIZE_BYTES * 2 != FULL_SIZE_BYTES {
        panic!("what?");
    }
};
#[derive(Clone, Copy)]
pub union HalfSize {
    ne_bytes: [u8; HALF_SIZE_BYTES],
    native: HalfSizeNative,
}
impl std::fmt::Debug for HalfSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HalfSize").field("native", &**self).finish()
    }
}
impl std::fmt::Display for HalfSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", &**self)
    }
}
impl std::fmt::LowerHex for HalfSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::LowerHex::fmt(
            &if cfg!(target_endian = "little") {
                **self
            } else {
                HalfSizeNative::from_be(**self)
            },
            f,
        )
    }
}
impl std::fmt::UpperHex for HalfSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::UpperHex::fmt(
            &if cfg!(target_endian = "little") {
                **self
            } else {
                HalfSizeNative::from_be(**self)
            },
            f,
        )
    }
}

impl PartialEq for HalfSize {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}
impl Eq for HalfSize {}
impl PartialOrd for HalfSize {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HalfSize {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (**self).cmp(&**other)
    }
}

impl Default for HalfSize {
    fn default() -> Self {
        Self { native: 0 }
    }
}
impl From<HalfSizeNative> for HalfSize {
    fn from(value: HalfSizeNative) -> Self {
        Self::new(value)
    }
}
impl From<[u8; HALF_SIZE_BYTES]> for HalfSize {
    fn from(value: [u8; HALF_SIZE_BYTES]) -> Self {
        Self { ne_bytes: value }
    }
}

impl HalfSize {
    pub const fn new(native: HalfSizeNative) -> Self {
        Self { native }
    }
    fn format_index(index: usize) -> usize {
        assert!(index < HALF_SIZE_BYTES);
        if cfg!(target_endian = "little") {
            index
        } else {
            HALF_SIZE_BYTES - index
        }
    }
    pub const fn ne_bytes(self) -> [u8; HALF_SIZE_BYTES] {
        // SAFETY: union will always be properly initialized
        unsafe { self.ne_bytes }
    }
    pub fn le_bytes(self) -> [u8; HALF_SIZE_BYTES] {
        (*self).to_le_bytes()
    }
    pub fn be_bytes(self) -> [u8; HALF_SIZE_BYTES] {
        (*self).to_be_bytes()
    }
}
impl std::ops::Deref for HalfSize {
    type Target = HalfSizeNative;

    fn deref(&self) -> &Self::Target {
        // SAFETY: union will always be properly initialized
        unsafe { &self.native }
    }
}
impl std::ops::DerefMut for HalfSize {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: union will always be properly initialized
        unsafe { &mut self.native }
    }
}

/// access le ordered bytes
impl std::ops::Index<usize> for HalfSize {
    type Output = u8;

    fn index(&self, index: usize) -> &Self::Output {
        // SAFETY: union will always be properly initialized
        unsafe { self.ne_bytes.index(Self::format_index(index)) }
    }
}
/// access le ordered bytes
impl std::ops::IndexMut<usize> for HalfSize {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        // SAFETY: union will always be properly initialized
        unsafe { self.ne_bytes.index_mut(Self::format_index(index)) }
    }
}

#[derive(Clone, Copy)]
pub union FullSize {
    native: usize,
    halfs: [HalfSize; 2],
    #[allow(dead_code)]
    ne_bytes: [u8; FULL_SIZE_BYTES],
}
impl std::fmt::Debug for FullSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:#x}")
    }
}
impl std::fmt::Display for FullSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", **self)
    }
}
impl std::fmt::LowerHex for FullSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::LowerHex::fmt(
            &if cfg!(target_endian = "little") {
                **self
            } else {
                usize::from_be(**self)
            },
            f,
        )
    }
}
impl std::fmt::UpperHex for FullSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::UpperHex::fmt(
            &if cfg!(target_endian = "little") {
                **self
            } else {
                usize::from_be(**self)
            },
            f,
        )
    }
}

impl PartialEq for FullSize {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}
impl Eq for FullSize {}

impl From<usize> for FullSize {
    fn from(native: usize) -> Self {
        // SAFTY: access to native is always possible
        Self { native }
    }
}
impl From<HalfSize> for FullSize {
    fn from(lower: HalfSize) -> Self {
        Self::new(lower, HalfSize::default())
    }
}
/// SAFTY: access to part is always possible
#[allow(clippy::undocumented_unsafe_blocks)]
impl FullSize {
    pub const fn new(lower: HalfSize, higher: HalfSize) -> Self {
        if cfg!(target_endian = "little") {
            Self {
                halfs: [lower, higher],
            }
        } else {
            Self {
                halfs: [higher, lower],
            }
        }
    }
    pub const fn lower(self) -> HalfSize {
        if cfg!(target_endian = "little") {
            unsafe { self.halfs[0] }
        } else {
            unsafe { self.halfs[1] }
        }
    }
    pub const fn higher(self) -> HalfSize {
        if cfg!(target_endian = "little") {
            unsafe { self.halfs[1] }
        } else {
            unsafe { self.halfs[0] }
        }
    }
}

impl std::ops::Deref for FullSize {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        // SAFETY: union will always be properly initialized
        unsafe { &self.native }
    }
}
impl std::ops::DerefMut for FullSize {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: union will always be properly initialized
        unsafe { &mut self.native }
    }
}

macro_rules! implHalfMath {
    (a $($trait:tt)::*, $func:tt) => {
        implHalfMath!(a $($trait)::*, $func, Self);
        implHalfMath!(a $($trait)::*, $func, &Self);
        implHalfMath!(a $($trait)::*, $func, u32);
        implHalfMath!(a $($trait)::*, $func, &u32);
    };
    (a $($trait:tt)::*, $func:tt, Self) => {
        impl $($trait)::* for HalfSize {
            fn $func(&mut self, rhs: Self) {
                $($trait)::*::$func(&mut **self, *rhs)
            }
        }
    };
    (a $($trait:tt)::*, $func:tt, &Self) => {
        impl $($trait)::*<&Self> for HalfSize {
            fn $func(&mut self, rhs: &Self) {
                $($trait)::*::$func(&mut **self, **rhs)
            }
        }
    };
    (a $($trait:tt)::*, $func:tt, $rhs:tt) => {
        impl $($trait)::*<$rhs> for HalfSize {
            fn $func(&mut self, rhs: $rhs) {
                $($trait)::*::$func(&mut **self, rhs)
            }
        }
    };
    (a $($trait:tt)::*, $func:tt, &$rhs:tt) => {
        impl $($trait)::*<&$rhs> for HalfSize {
            fn $func(&mut self, rhs: &$rhs) {
                $($trait)::*::$func(&mut **self, *rhs)
            }
        }
    };

    ($($trait:tt)::*, $func:tt) => {
        implHalfMath!($($trait)::*, $func, Self);
        implHalfMath!($($trait)::*, $func, &Self);
        implHalfMath!($($trait)::*, $func, u32);
        implHalfMath!($($trait)::*, $func, &u32);
    };
    ($($trait:tt)::*, $func:tt, Self) => {
        impl $($trait)::* for HalfSize {
            type Output = Self;
            fn $func(self, rhs: Self) -> Self::Output  {
                Self::from($($trait)::*::$func(*self, *rhs))
            }
        }
        impl $($trait)::* for &HalfSize {
            type Output = HalfSize;
            fn $func(self, rhs: Self) -> Self::Output  {
                $($trait)::*::$func(*self, *rhs)
            }
        }
    };
    ($($trait:tt)::*, $func:tt, &Self) => {
        impl $($trait)::*<&Self> for HalfSize {
            type Output = Self;
            fn $func(self, rhs: &Self) -> Self::Output  {
                Self::from($($trait)::*::$func(*self, **rhs))
            }
        }
        impl $($trait)::*<&Self> for &HalfSize {
            type Output = HalfSize;
            fn $func(self, rhs: &Self) -> Self::Output  {
                $($trait)::*::$func(*self, **rhs)
            }
        }
    };
    ($($trait:tt)::*, $func:tt, $rhs:tt) => {
        impl $($trait)::*<$rhs> for HalfSize {
            type Output = Self;
            fn $func(self, rhs: $rhs) -> Self::Output  {
                Self::from($($trait)::*::$func( *self, rhs))
            }
        }
        impl $($trait)::*<$rhs> for &HalfSize {
            type Output = HalfSize;
            fn $func(self, rhs: $rhs) -> Self::Output  {
                $($trait)::*::$func(*self, rhs)
            }
        }
    };
    ($($trait:tt)::*, $func:tt, &$rhs:tt) => {
        impl $($trait)::*<&$rhs> for HalfSize {
            type Output = Self;
            fn $func(self, rhs: &$rhs) -> Self::Output  {
                Self::from($($trait)::*::$func( *self, *rhs))
            }
        }
        impl $($trait)::*<&$rhs> for &HalfSize {
            type Output = HalfSize;
            fn $func(self, rhs: &$rhs) -> Self::Output  {
                $($trait)::*::$func(*self, *rhs)
            }
        }
    };
}
implHalfMath!(a std::ops::BitOrAssign, bitor_assign);
implHalfMath!(std::ops::BitOr, bitor);
implHalfMath!(a std::ops::BitXorAssign, bitxor_assign);
implHalfMath!(std::ops::BitXor, bitxor);
implHalfMath!(a std::ops::BitAndAssign, bitand_assign);
implHalfMath!(std::ops::BitAnd, bitand);

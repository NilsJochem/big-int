use std::ops::{Deref, DerefMut};

/// Borrowed or Owned, used to capture all possible variants when implementing traits for Self/&Self
#[derive(Debug, derive_more::From)]
pub enum Boo<'b, T> {
    Owned(T),
    Borrowed(&'b T),
    BorrowedMut(&'b mut T),
}

impl<'b, T> From<Moo<'b, T>> for Boo<'b, T> {
    fn from(value: Moo<'b, T>) -> Self {
        match value {
            Moo::Owned(owned) => Boo::Owned(owned),
            Moo::BorrowedMut(borrow) => Boo::BorrowedMut(borrow),
        }
    }
}

impl<'b, T> Boo<'b, T> {
    /// gives an owned instance of `T` by using `deref` on the held reference
    pub fn into_owned(self, deref: impl FnOnce(&'b T) -> T) -> T {
        match self {
            Boo::Owned(t) => t,
            Boo::Borrowed(t) => deref(t),
            Boo::BorrowedMut(t) => deref(t),
        }
    }

    pub fn try_get_mut(&mut self) -> Option<&mut T> {
        match self {
            Boo::Owned(t) => Some(t),
            Boo::BorrowedMut(t) => Some(t),
            Boo::Borrowed(_) => None,
        }
    }

    /// gives an owned instance of `T` by cloning the held reference
    pub fn cloned(self) -> T
    where
        T: Clone,
    {
        self.into_owned(T::clone)
    }
}
impl<'b, T> Deref for Boo<'b, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match self {
            Boo::Owned(t) => t,
            Boo::Borrowed(t) => t,
            Boo::BorrowedMut(t) => t,
        }
    }
}
impl<'b, T> AsRef<T> for Boo<'b, T> {
    fn as_ref(&self) -> &T {
        match self {
            Boo::Owned(t) => t,
            Boo::Borrowed(t) => t,
            Boo::BorrowedMut(t) => t,
        }
    }
}

/// Mutable referenec Or Owned
#[derive(Debug, derive_more::From)]
pub enum Moo<'b, T> {
    Owned(T),
    BorrowedMut(&'b mut T),
}

impl<'b, T: Clone> From<Boo<'b, T>> for Moo<'b, T> {
    fn from(value: Boo<'b, T>) -> Self {
        match value {
            Boo::BorrowedMut(value) => Moo::BorrowedMut(value),
            value => Moo::Owned(value.cloned()),
        }
    }
}
impl<'b, T> Moo<'b, T> {
    pub fn expect_owned(self, msg: impl AsRef<str>) -> T {
        #[allow(clippy::expect_fun_call)]
        self.try_get_owned().expect(msg.as_ref())
    }
    pub fn try_get_owned(self) -> Option<T> {
        match self {
            Moo::Owned(it) => Some(it),
            Moo::BorrowedMut(_) => None,
        }
    }
    pub fn expect_mut_ref(self, msg: impl AsRef<str>) -> &'b mut T {
        #[allow(clippy::expect_fun_call)]
        self.try_get_mut_ref().expect(msg.as_ref())
    }
    pub fn try_get_mut_ref(self) -> Option<&'b mut T> {
        match self {
            Moo::Owned(_) => None,
            Moo::BorrowedMut(it) => Some(it),
        }
    }
}
impl<'b, T> Deref for Moo<'b, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match self {
            Moo::Owned(it) => it,
            Moo::BorrowedMut(it) => it,
        }
    }
}
impl<'b, T> DerefMut for Moo<'b, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Moo::Owned(it) => it,
            Moo::BorrowedMut(it) => it,
        }
    }
}

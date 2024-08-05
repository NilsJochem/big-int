use std::ops::Deref;

/// Borrowed or Owned, used to capture all possible variants when implementing traits for Self/&Self
#[derive(Debug, derive_more::From)]
pub enum Boo<'b, T> {
    Owned(T),
    Borrowed(&'b T),
    BorrowedMut(&'b mut T),
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

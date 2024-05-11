use core::iter::{once, Chain, Once};

/// An iterator over two items
pub type Chain2<T> = Chain<Once<T>, Once<T>>;
/// An iterator over three items
pub type Chain3<T> = Chain<Chain<Once<T>, Once<T>>, Once<T>>;

/**
Trait for defining a pair of items of the same type.

This trait is meant to generalize having two similar things.
It is implemented for `(T, T)` and `[T; 2]` with `Item = T`.
However, because a pair does not necessarily have to be an
actual *pair* It is also implemented for `(T, T, T, T)` and
`[T; 4]` with `Item = (T, T)` and `Item = [T; 2]` respectively.
*/
pub trait Pair: Sized {
    /// The type of the pair's item
    type Item;
    /// Get the pair
    fn into_pair(self) -> (Self::Item, Self::Item);
    /// Create a pair from two items
    fn from_items(a: Self::Item, b: Self::Item) -> Self;
    /// Get the first item
    fn first(&self) -> Self::Item;
    /// Get the second item
    fn second(&self) -> Self::Item;
    /// Apply a function pairwise to the items of two pairs
    fn pairwise<O, P, F, R>(self, other: O, f: F) -> P
    where
        O: Pair,
        P: Pair<Item = R>,
        F: Fn(Self::Item, O::Item) -> R,
    {
        let (a, b) = self.into_pair();
        let (c, d) = other.into_pair();
        P::from_items(f(a, c), f(b, d))
    }
    /// Get an iterator over the pair's items
    fn pair_iter(self) -> Chain2<Self::Item> {
        let (a, b) = self.into_pair();
        once(a).chain(once(b))
    }
}

impl<T> Pair for (T, T)
where
    T: Clone,
{
    type Item = T;
    #[inline(always)]
    fn into_pair(self) -> (Self::Item, Self::Item) {
        self
    }
    #[inline(always)]
    fn from_items(a: Self::Item, b: Self::Item) -> Self {
        (a, b)
    }
    #[inline(always)]
    fn first(&self) -> Self::Item {
        self.0.clone()
    }
    #[inline(always)]
    fn second(&self) -> Self::Item {
        self.1.clone()
    }
}

impl<T> Pair for [T; 2]
where
    T: Copy,
{
    type Item = T;
    #[inline(always)]
    fn into_pair(self) -> (Self::Item, Self::Item) {
        (self[0], self[1])
    }
    #[inline(always)]
    fn from_items(a: Self::Item, b: Self::Item) -> Self {
        [a, b]
    }
    #[inline(always)]
    fn first(&self) -> Self::Item {
        self[0]
    }
    #[inline(always)]
    fn second(&self) -> Self::Item {
        self[1]
    }
}

impl<T> Pair for (T, T, T, T)
where
    T: Clone,
{
    type Item = (T, T);
    #[inline(always)]
    fn into_pair(self) -> (Self::Item, Self::Item) {
        ((self.0, self.1), (self.2, self.3))
    }
    #[inline(always)]
    fn from_items(a: Self::Item, b: Self::Item) -> Self {
        (a.0, a.1, b.0, b.1)
    }
    #[inline(always)]
    fn first(&self) -> Self::Item {
        (self.0.clone(), self.1.clone())
    }
    #[inline(always)]
    fn second(&self) -> Self::Item {
        (self.2.clone(), self.3.clone())
    }
}

impl<T> Pair for [T; 4]
where
    T: Copy,
{
    type Item = [T; 2];
    #[inline(always)]
    fn into_pair(self) -> (Self::Item, Self::Item) {
        ([self[0], self[1]], [self[2], self[3]])
    }
    #[inline(always)]
    fn from_items(a: Self::Item, b: Self::Item) -> Self {
        [a[0], a[1], b[0], b[1]]
    }
    #[inline(always)]
    fn first(&self) -> Self::Item {
        [self[0], self[1]]
    }
    #[inline(always)]
    fn second(&self) -> Self::Item {
        [self[2], self[3]]
    }
}

impl<T> Pair for (T, T, T, T, T, T)
where
    T: Clone,
{
    type Item = (T, T, T);
    #[inline(always)]
    fn into_pair(self) -> (Self::Item, Self::Item) {
        ((self.0, self.1, self.2), (self.3, self.4, self.5))
    }
    #[inline(always)]
    fn from_items(a: Self::Item, b: Self::Item) -> Self {
        (a.0, a.1, a.2, b.0, b.1, b.2)
    }
    #[inline(always)]
    fn first(&self) -> Self::Item {
        (self.0.clone(), self.1.clone(), self.2.clone())
    }
    #[inline(always)]
    fn second(&self) -> Self::Item {
        (self.3.clone(), self.4.clone(), self.5.clone())
    }
}

impl<T> Pair for [T; 6]
where
    T: Copy,
{
    type Item = [T; 3];
    #[inline(always)]
    fn into_pair(self) -> (Self::Item, Self::Item) {
        ([self[0], self[1], self[2]], [self[3], self[4], self[5]])
    }
    #[inline(always)]
    fn from_items(a: Self::Item, b: Self::Item) -> Self {
        [a[0], a[1], a[2], b[0], b[1], b[2]]
    }
    #[inline(always)]
    fn first(&self) -> Self::Item {
        [self[0], self[1], self[2]]
    }
    #[inline(always)]
    fn second(&self) -> Self::Item {
        [self[3], self[4], self[5]]
    }
}

/**
Trait for defining a group of 3 items of the same type.

This trait is meant to generalize having three similar things.
It is implemented for `(T, T, T)` and `[T; 3]` with `Item = T`.
However, because a trio does not necessarily have to be an
actual tuple It is also implemented for `(T, T, T, T, T, T)` and
`[T; 6]` with `Item = (T, T, T)` and `Item = [T; 3]` respectively.
*/

pub trait Trio: Sized {
    /// The type of the trio's item
    type Item;
    /// Get the trio
    #[allow(clippy::wrong_self_convention)]
    fn into_trio(self) -> (Self::Item, Self::Item, Self::Item);
    /// Create a trio from three items
    fn from_items(a: Self::Item, b: Self::Item, c: Self::Item) -> Self;
    /// Apply a function pairwise to the items of two trios
    fn pairwise<O, T, F, R>(self, other: O, ff: F) -> T
    where
        O: Trio,
        T: Trio<Item = R>,
        F: Fn(Self::Item, O::Item) -> R,
    {
        let (a, b, c) = self.into_trio();
        let (d, e, f) = other.into_trio();
        T::from_items(ff(a, d), ff(b, e), ff(c, f))
    }
    /// Get an iterator over the trio's items
    fn trio_iter(self) -> Chain3<Self::Item> {
        let (a, b, c) = self.into_trio();
        once(a).chain(once(b)).chain(once(c))
    }
}

impl<T> Trio for (T, T, T) {
    type Item = T;
    #[inline(always)]
    fn into_trio(self) -> (Self::Item, Self::Item, Self::Item) {
        self
    }
    #[inline(always)]
    fn from_items(a: Self::Item, b: Self::Item, c: Self::Item) -> Self {
        (a, b, c)
    }
}

impl<T> Trio for [T; 3]
where
    T: Copy,
{
    type Item = T;
    #[inline(always)]
    fn into_trio(self) -> (Self::Item, Self::Item, Self::Item) {
        (self[0], self[1], self[2])
    }
    #[inline(always)]
    fn from_items(a: Self::Item, b: Self::Item, c: Self::Item) -> Self {
        [a, b, c]
    }
}

impl<T> Trio for (T, T, T, T, T, T) {
    type Item = (T, T);
    #[inline(always)]
    fn into_trio(self) -> (Self::Item, Self::Item, Self::Item) {
        ((self.0, self.1), (self.2, self.3), (self.4, self.5))
    }
    #[inline(always)]
    fn from_items(a: Self::Item, b: Self::Item, c: Self::Item) -> Self {
        (a.0, a.1, b.0, b.1, c.0, c.1)
    }
}

impl<T> Trio for [T; 6]
where
    T: Copy,
{
    type Item = [T; 2];
    #[inline(always)]
    fn into_trio(self) -> (Self::Item, Self::Item, Self::Item) {
        ([self[0], self[1]], [self[2], self[3]], [self[4], self[5]])
    }
    #[inline(always)]
    fn from_items(a: Self::Item, b: Self::Item, c: Self::Item) -> Self {
        [a[0], a[1], b[0], b[1], c[0], c[1]]
    }
}

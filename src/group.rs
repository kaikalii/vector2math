/// Trait for defining a pair of items of the same type.
///
/// This trait is meant to generalize having two similar things.
/// It is implemented for `(T, T)` and `[T; 2]` with `Item = T`.
/// However, because a pair does not necessarily have to be an
/// actual *pair* It is also implemented for `(T, T, T, T)` and
/// `[T; 4]` with `Item = (T, T)` and `Item = [T; 2]` respectively.
pub trait Pair {
    /// The type of the pair's item
    type Item;
    /// Get the first thing
    fn first(self) -> Self::Item;
    /// Get the second thing
    fn second(self) -> Self::Item;
    /// Create a pair from two items
    fn from_items(a: Self::Item, b: Self::Item) -> Self;
}

impl<T> Pair for (T, T)
where
    T: Clone,
{
    type Item = T;
    fn first(self) -> Self::Item {
        self.0
    }
    fn second(self) -> Self::Item {
        self.1
    }
    fn from_items(a: Self::Item, b: Self::Item) -> Self {
        (a, b)
    }
}

impl<T> Pair for [T; 2]
where
    T: Clone,
{
    type Item = T;
    fn first(self) -> Self::Item {
        self[0].clone()
    }
    fn second(self) -> Self::Item {
        self[1].clone()
    }
    fn from_items(a: Self::Item, b: Self::Item) -> Self {
        [a, b]
    }
}

impl<T> Pair for (T, T, T, T)
where
    T: Clone,
{
    type Item = (T, T);
    fn first(self) -> Self::Item {
        (self.0, self.1)
    }
    fn second(self) -> Self::Item {
        (self.2, self.3)
    }
    fn from_items(a: Self::Item, b: Self::Item) -> Self {
        (a.0, a.1, b.0, b.1)
    }
}

impl<T> Pair for [T; 4]
where
    T: Clone,
{
    type Item = [T; 2];
    fn first(self) -> Self::Item {
        [self[0].clone(), self[1].clone()]
    }
    fn second(self) -> Self::Item {
        [self[2].clone(), self[3].clone()]
    }
    fn from_items(a: Self::Item, b: Self::Item) -> Self {
        [a[0].clone(), a[1].clone(), b[0].clone(), b[1].clone()]
    }
}

/// Trait for defining a group of 3 items of the same type.
///
/// This trait is meant to generalize having three similar things.
/// It is implemented for `(T, T, T)` and `[T; 3]` with `Item = T`.
/// However, because a trio does not necessarily have to be an
/// actual tuple It is also implemented for `(T, T, T, T, T, T)` and
/// `[T; 6]` with `Item = (T, T, T)` and `Item = [T; 3]` respectively.

pub trait Trio {
    /// The type of the trio's item
    type Item;
    /// Get the first thing
    fn first(self) -> Self::Item;
    /// Get the second thing
    fn second(self) -> Self::Item;
    /// Get the third thing
    fn third(self) -> Self::Item;
    /// Create a trio from three items
    fn from_items(a: Self::Item, b: Self::Item, c: Self::Item) -> Self;
}

impl<T> Trio for (T, T, T)
where
    T: Clone,
{
    type Item = T;
    fn first(self) -> Self::Item {
        self.0
    }
    fn second(self) -> Self::Item {
        self.1
    }
    fn third(self) -> Self::Item {
        self.2
    }
    fn from_items(a: Self::Item, b: Self::Item, c: Self::Item) -> Self {
        (a, b, c)
    }
}

impl<T> Trio for [T; 3]
where
    T: Clone,
{
    type Item = T;
    fn first(self) -> Self::Item {
        self[0].clone()
    }
    fn second(self) -> Self::Item {
        self[1].clone()
    }
    fn third(self) -> Self::Item {
        self[1].clone()
    }
    fn from_items(a: Self::Item, b: Self::Item, c: Self::Item) -> Self {
        [a, b, c]
    }
}

impl<T> Trio for (T, T, T, T, T, T)
where
    T: Clone,
{
    type Item = (T, T);
    fn first(self) -> Self::Item {
        (self.0, self.1)
    }
    fn second(self) -> Self::Item {
        (self.2, self.3)
    }
    fn third(self) -> Self::Item {
        (self.4, self.5)
    }
    fn from_items(a: Self::Item, b: Self::Item, c: Self::Item) -> Self {
        (a.0, a.1, b.0, b.1, c.0, c.1)
    }
}

impl<T> Trio for [T; 6]
where
    T: Clone,
{
    type Item = [T; 2];
    fn first(self) -> Self::Item {
        [self[0].clone(), self[1].clone()]
    }
    fn second(self) -> Self::Item {
        [self[2].clone(), self[3].clone()]
    }
    fn third(self) -> Self::Item {
        [self[4].clone(), self[5].clone()]
    }
    fn from_items(a: Self::Item, b: Self::Item, c: Self::Item) -> Self {
        [
            a[0].clone(),
            a[1].clone(),
            b[0].clone(),
            b[1].clone(),
            c[0].clone(),
            c[1].clone(),
        ]
    }
}

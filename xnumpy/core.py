__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Oct 12, 2016 16:22$"


import numpy


def expand(new_array,
           shape_after=tuple(),
           shape_before=tuple(),
           read_only=True):
    """
        Tack on extra dimensions of the specified size to either side.

        Behaves like NumPy tile except that it always returns a view not a
        copy. Though, it differs in that additional dimensions are added for
        repetition as opposed to repeating in the same one. Also, it allows
        repetitions to be specified before or after unlike tile. Though, will
        behave identical to tile if the keyword is not specified.
        Uses strides to trick NumPy into providing a view.

        Args:
            new_array(numpy.ndarray):            array to expand

            shape_after(tuple or int):           size of dimensions to add
                                                 before the array shape (if
                                                 int will turn into tuple).
                                                 Defaults to an empty tuple.

            shape_before(tuple or int):          size of dimensions to add
                                                 before the array shape (if
                                                 int will turn into tuple).
                                                 Defaults to an empty tuple.

            read_only(bool):
        Returns:
            (numpy.ndarray):                     a view of a numpy array with
                                                 tiling in various dimension.

        Examples:
            >>> numpy.arange(6).reshape(2,3)
            array([[0, 1, 2],
                   [3, 4, 5]])

            >>> expand(numpy.arange(6).reshape(2,3))
            array([[0, 1, 2],
                   [3, 4, 5]])

            >>> a = numpy.arange(6).reshape(2,3); a is expand(a)
            False

            >>> expand(numpy.arange(6).reshape(2,3), 1)
            array([[[0],
                    [1],
                    [2]],
            <BLANKLINE>
                   [[3],
                    [4],
                    [5]]])

            >>> expand(numpy.arange(6).reshape(2,3), (1,))
            array([[[0],
                    [1],
                    [2]],
            <BLANKLINE>
                   [[3],
                    [4],
                    [5]]])

            >>> expand(numpy.arange(6).reshape((2,3)), shape_after=1)
            array([[[0],
                    [1],
                    [2]],
            <BLANKLINE>
                   [[3],
                    [4],
                    [5]]])

            >>> expand(numpy.arange(6).reshape((2,3)), shape_after=(1,))
            array([[[0],
                    [1],
                    [2]],
            <BLANKLINE>
                   [[3],
                    [4],
                    [5]]])

            >>> expand(numpy.arange(6).reshape((2,3)), shape_before=1)
            array([[[0, 1, 2],
                    [3, 4, 5]]])

            >>> expand(numpy.arange(6).reshape((2,3)), shape_before=(1,))
            array([[[0, 1, 2],
                    [3, 4, 5]]])

            >>> expand(numpy.arange(6).reshape((2,3)), shape_before=(3,))
            array([[[0, 1, 2],
                    [3, 4, 5]],
            <BLANKLINE>
                   [[0, 1, 2],
                    [3, 4, 5]],
            <BLANKLINE>
                   [[0, 1, 2],
                    [3, 4, 5]]])

            >>> expand(numpy.arange(6).reshape((2,3)), shape_after=(4,))
            array([[[0, 0, 0, 0],
                    [1, 1, 1, 1],
                    [2, 2, 2, 2]],
            <BLANKLINE>
                   [[3, 3, 3, 3],
                    [4, 4, 4, 4],
                    [5, 5, 5, 5]]])

            >>> expand(
            ...     numpy.arange(6).reshape((2,3)),
            ...     shape_before=(3,),
            ...     shape_after=(4,)
            ... )
            array([[[[0, 0, 0, 0],
                     [1, 1, 1, 1],
                     [2, 2, 2, 2]],
            <BLANKLINE>
                    [[3, 3, 3, 3],
                     [4, 4, 4, 4],
                     [5, 5, 5, 5]]],
            <BLANKLINE>
            <BLANKLINE>
                   [[[0, 0, 0, 0],
                     [1, 1, 1, 1],
                     [2, 2, 2, 2]],
            <BLANKLINE>
                    [[3, 3, 3, 3],
                     [4, 4, 4, 4],
                     [5, 5, 5, 5]]],
            <BLANKLINE>
            <BLANKLINE>
                   [[[0, 0, 0, 0],
                     [1, 1, 1, 1],
                     [2, 2, 2, 2]],
            <BLANKLINE>
                    [[3, 3, 3, 3],
                     [4, 4, 4, 4],
                     [5, 5, 5, 5]]]])

            >>> expand(numpy.arange(6).reshape((2,3)), shape_after = (4,3))
            array([[[[0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0]],
            <BLANKLINE>
                    [[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 1]],
            <BLANKLINE>
                    [[2, 2, 2],
                     [2, 2, 2],
                     [2, 2, 2],
                     [2, 2, 2]]],
            <BLANKLINE>
            <BLANKLINE>
                   [[[3, 3, 3],
                     [3, 3, 3],
                     [3, 3, 3],
                     [3, 3, 3]],
            <BLANKLINE>
                    [[4, 4, 4],
                     [4, 4, 4],
                     [4, 4, 4],
                     [4, 4, 4]],
            <BLANKLINE>
                    [[5, 5, 5],
                     [5, 5, 5],
                     [5, 5, 5],
                     [5, 5, 5]]]])

            >>> expand(
            ...     numpy.arange(6).reshape((2,3)),
            ...     shape_before=(4,3),
            ... )
            array([[[[0, 1, 2],
                     [3, 4, 5]],
            <BLANKLINE>
                    [[0, 1, 2],
                     [3, 4, 5]],
            <BLANKLINE>
                    [[0, 1, 2],
                     [3, 4, 5]]],
            <BLANKLINE>
            <BLANKLINE>
                   [[[0, 1, 2],
                     [3, 4, 5]],
            <BLANKLINE>
                    [[0, 1, 2],
                     [3, 4, 5]],
            <BLANKLINE>
                    [[0, 1, 2],
                     [3, 4, 5]]],
            <BLANKLINE>
            <BLANKLINE>
                   [[[0, 1, 2],
                     [3, 4, 5]],
            <BLANKLINE>
                    [[0, 1, 2],
                     [3, 4, 5]],
            <BLANKLINE>
                    [[0, 1, 2],
                     [3, 4, 5]]],
            <BLANKLINE>
            <BLANKLINE>
                   [[[0, 1, 2],
                     [3, 4, 5]],
            <BLANKLINE>
                    [[0, 1, 2],
                     [3, 4, 5]],
            <BLANKLINE>
                    [[0, 1, 2],
                     [3, 4, 5]]]])
    """

    if not isinstance(shape_after, tuple):
        shape_after = (shape_after,)

    if not isinstance(shape_before, tuple):
        shape_before = (shape_before,)

    new_array_expanded = numpy.lib.stride_tricks.as_strided(
        new_array,
        shape_before + new_array.shape + shape_after,
        (0,) * len(shape_before) + new_array.strides + (0,) * len(shape_after)
    )
    new_array_expanded.flags["WRITEABLE"] = not read_only

    return(new_array_expanded)


def enumerate(new_array, axis=0, start=0, step=1, dtype=None):
    """
        Builds on expand_arange, which has the same shape as the original
        array. Specifies the increments to occur along the given axis, which by
        default is the zeroth axis.
        Provides mechanisms for changing the starting value and also the
        increment.
        Args:
            new_array(numpy.ndarray):            array to enumerate
            start(int):                          starting point (0 by default).
            step(int):                           size of steps to take between
                                                 value (1 by default).
            axis(int):                           axis to enumerate along (0 by
                                                 default).
            dtype(type):                         type to use for enumerating.

        Returns:
            (numpy.ndarray):                     a view of a numpy arange with
                                                 tiling in various dimension.
        Examples:
            >>> enumerate(
            ...     numpy.ones((4,5), dtype=numpy.uint64)
            ... )
            array([[0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1],
                   [2, 2, 2, 2, 2],
                   [3, 3, 3, 3, 3]], dtype=uint64)

            >>> enumerate(
            ...     numpy.ones((4,5), dtype=numpy.uint64),
            ...     axis=0
            ... )
            array([[0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1],
                   [2, 2, 2, 2, 2],
                   [3, 3, 3, 3, 3]], dtype=uint64)

            >>> enumerate(
            ...     numpy.ones((4,5), dtype=numpy.uint64),
            ...     axis=0,
            ...     start=1
            ... )
            array([[1, 1, 1, 1, 1],
                   [2, 2, 2, 2, 2],
                   [3, 3, 3, 3, 3],
                   [4, 4, 4, 4, 4]], dtype=uint64)

            >>> enumerate(
            ...     numpy.ones((4,5), dtype=numpy.uint64),
            ...     axis=0,
            ...     start=1,
            ...     step=2
            ... )
            array([[1, 1, 1, 1, 1],
                   [3, 3, 3, 3, 3],
                   [5, 5, 5, 5, 5],
                   [7, 7, 7, 7, 7]], dtype=uint64)

            >>> enumerate(
            ...     numpy.ones((4,5), dtype=numpy.uint64),
            ...     axis=1
            ... )
            array([[0, 1, 2, 3, 4],
                   [0, 1, 2, 3, 4],
                   [0, 1, 2, 3, 4],
                   [0, 1, 2, 3, 4]], dtype=uint64)

            >>> enumerate(
            ...     numpy.ones((4,5), dtype=numpy.uint64),
            ...     axis=1,
            ...     start=1
            ... )
            array([[1, 2, 3, 4, 5],
                   [1, 2, 3, 4, 5],
                   [1, 2, 3, 4, 5],
                   [1, 2, 3, 4, 5]], dtype=uint64)

            >>> enumerate(
            ...     numpy.ones((4,5), dtype=numpy.uint64),
            ...     axis=1,
            ...     start=1,
            ...     step=2
            ... )
            array([[1, 3, 5, 7, 9],
                   [1, 3, 5, 7, 9],
                   [1, 3, 5, 7, 9],
                   [1, 3, 5, 7, 9]], dtype=uint64)
    """

    if dtype is None:
        dtype = new_array.dtype

    dtype = numpy.dtype(dtype)

    an_enumeration = expand(
        numpy.arange(
            start=start,
            stop=start + step * new_array.shape[axis],
            step=step,
            dtype=dtype
        ),
        shape_before=new_array.shape[:axis],
        shape_after=new_array.shape[(axis+1):]
    )

    return(an_enumeration)

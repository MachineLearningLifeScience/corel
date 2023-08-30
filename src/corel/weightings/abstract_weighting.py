__author__ = 'Simon Bartels'


class AbstractWeighting:
    def expectation(self, p):
        """
        This function takes the expectation of p over all protein sequences w.r.t. a base distribution.
        The function p is assumed to be factorizing, i.e. p(x)=product_i(x[i]).
        An important special case: p may have mass on only one sequence.
        :param p:
            the function for which to take the expectation for
            The function p may have dtype integer and is then assumed to have mass on only one sequence.
        :return:
            The expectation of p w.r.t. a base distribution p_0: sum_x p(x) p_0(x)
        """
        raise NotImplementedError("abstract method")

    def __call__(self, *args, **kwargs):
        return self.expectation(*args, **kwargs)

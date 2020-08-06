The "heat equation" problem.
In the following, I address a few remarks concerning implementation.


# PART 1
* Most mathematical functions concerning vectors were kept outside of the Vector class, including the dot product. This happend so that I could allow using vectors of different type (ex. int and double).
* I added more mathematical functions to have a more complete functionality.
* In addition, I added more class functions for comparing two vectors based on a tolerance. One function for comparing only the element-wise difference and another to compare the squared norm to the aforementioned tolerance.

# PART 2
* In the `matvec` method in `Matrix` class, I implemented a sparse matrix-vector multiplication based on the COO format that is facilitated by the key `array<int, 2>` of the map.

# PART 3
I slightly changed the algorithm of the pseudocode to increase efficiency at the cost of memory.
* 1 vector of length N to store the result of A.matvec(p_k) to avoid computing it twice.
* 2 scalars for the squared norms. Like that, I calculate each squared norm once instead of thrice.

# PART 4
* U renamed the `template<int n> class Heat` into `template<int n> class HeatnD`
* All Heat1D, Heat2D and HeatnD classes were made. I started with the first 2 unsure of whether there would be enough time for the template class.
* Almost duplicate code exists between these classes, but moving the specific functions outside of the classes seemed counter-intuitive.
* Functions `print_solution` and `save_solution` do what their titles suggest in COO format. Therefore, the distance between each node is `1` arbitrary unit and no `dx`. In addition, the homogeneous boundary conditions are not included. The decision of how to address these issues is left to the user.

# REMARKS
* One file and Makefile for simplicity.
* C ++14 standards used.
* By profiling with `-pg`, we can see information such as that the `matvec` function takes up most of the computation time. This shows the importance of the sparse matrix-vector multiplication.
* The result were confirmed by viewing print-outs on the screen and by saving the solutions on a file and then plotting them with Gnuplot.

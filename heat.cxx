/**
 * \file heat.cxx
 *
 *
 * Object Oriented Scientific Programming with C++
 *
 *
 *         AUTHOR NAME
 * \author Dilan Gecmn  
 *
 * More information in README.md
 *
 */
//////////////////////////////////////////////////////////////////////////////////



// Include header file for standard input/output stream library
#include <iostream>
// Include header file for file     input/output stream library
#include <fstream>

#include <initializer_list>
#include <memory>
#include <map>
#include <array>
#include <math.h>
#include <string>

//#include <typeinfo>


#define PI  3.141592653589793 // double accuracy
#define TOL 0.001
#define MAXIT 50


/////////////////////////////////////////////////////////////////////
//
//
//   TEMPLATE CLASS VECTOR
//
//
/////////////////////////////////////////////////////////////////////

template<typename T>
class Vector {
private:
  int length;
  T* data;

public:
  
  //
  //  CONSTRUCTORS
  // --------------
  //
  
  // Default constructor
  Vector():
    length(0),
    data(nullptr)
  {
    // std::cout << "Default constructor" << std::endl;
  }

  // Constructor (sets length and allocates data)
  Vector(const int length):
    length(length),
    data(new T[length])
  {
    // std::cout << "Length constructor" << std::endl;
  }

  // Constructor (sets length and allocates data to a specific value)
  Vector(const int length, T value):
    length(length),
    data(new T[length])
  {
    for (auto i = 0; i < length; ++i){
      data[i] = value;
    }
    // std::cout << "Length/value constructor" << std::endl;
  }

  // Copy Constructor
  //Vector(const Vector& c) = delete;
  Vector(const Vector& c):
    Vector(c.length)
  {
    for (auto i=0; i<c.length; i++)
      data[i] = c.data[i];

    // std::cout << "Copy constructor" << std::endl;
  }

  // Move Constructor
  //Vector(Vector&& c) = delete;
  Vector(Vector&& c):
    length(c.length),
    data(c.data)
  {
    c.length = 0;
    c.data = nullptr;
    
    // std::cout << "Move constructor" << std::endl;
  }

  // Constructor (using initializer list)
  Vector(std::initializer_list<T> list)
    : Vector((int)list.size())
  {
    std::uninitialized_copy(list.begin(), list.end(), data);
  }

  // Destructor
  ~Vector()
  {
    length=0;
    delete[] data;
  }


  //
  //  OPERATORS
  // -----------
  //

  // Returns the value of the key
  T& operator[](const int key)
  {
    return this->data[key];
  }

  // Returns the constant reference of value of the key
  // Does not allow changes
  const T& operator[](const int key) const
  {
    return this->data[key];
  }


  // Copy assignment
  Vector<T>& operator=(const Vector<T>& other)
  {
    if (this != &other)
      {
	delete[] data;
	length = other.length;
	data   = new T[other.length];
	for (auto i=0; i<other.length; i++)
	  data[i] = other.data[i];
      }
    
    // std::cout << "Copy assignment operator" << std::endl;
  
    return *this;
  }

  // Move assignment
  Vector<T>& operator=(Vector<T>&& other)
  {
    if (this != &other)
      {
	delete[] data;
	
	data   = other.data;
	length = other.length;

	other.data   = nullptr;
	other.length = 0;
      }
    
    // std::cout << "Move assignment operator" << std::endl;
  
    return *this;
  }
  

  //
  //  FUNCTIONS
  // -----------
  //
  
  // Returns the size of the container
  inline int size() const
  {
    return length;
  }

  // Returns the data of the container at the argument position
  inline T& get(const int i) const
  {
    return this->data[i];
  }
  
  // Returns the data of the container at the argument position
  void set(const int i, const T & value) const
  {
    this->data[i] = value;
  }

  bool compare(const Vector<T> & vec, T tol) const {

    int size = vec.size();

    if (this->length != size ) throw "Vectors have different size!";

    bool flag = true;

    for (auto i = 0; i < size; ++i){
      if ( fabs( this->data[i] - vec.data[i] ) > tol ){
	flag = false;
	std::cout << "Not equal at i = " << i << std::endl;
	std::cout << this->data[i] <<  " != " << vec.data[i] << std::endl;
	break;
      }
    }
    
    return flag;
  }
  
  bool diff_norm(const Vector<T> & vec, T tol){

    bool flag = true;
    int size = vec.size();

    if (this->length != size ) throw "Vectors have different size!";

    double norm = 0;
    for (auto i = 0; i < size; ++i){
      norm += pow(this->data[i] - vec.data[i], 2);
    }
    norm = sqrt(norm);

    if (norm > tol){
      flag = false;
    }
    
    std::cout << "Norm = " << norm << std::endl;
    
    return flag;
  }
  
};


/////////////////////////////////////////////////////////////////////
//
//
//   VECTOR RELATED TEMPLATE FUNCTIONS
//
//
/////////////////////////////////////////////////////////////////////



// ostream operator for `Vector`s
// Prints the length, followed by the data
template<typename T>
std::ostream &operator<<(std::ostream &os,
			 const Vector<T> &vec)
{  
  int size = vec.size();

  os << "Length = " << size << ", data";
  for (int i = 0; i < size; ++i){
    os << ", " << vec.get(i);
  }
  
  return os; 
}


// Vector-Vector addition (pointwise sum)
template<typename T1, typename T2>
auto operator+(const Vector<T1>& l,
	       const Vector<T2>& r)
{
  int size_l = l.size();
  int size_r = r.size();
  if (size_l != size_r) throw "Vectors have different size!";
  
  Vector<typename std::common_type<T1,T2>::type> result(size_l,0);
  for (auto i = 0; i < size_l; ++i)
    result.set(i, l.get(i) + r.get(i));
  //v.get[i] = v1.get[i] + v2.get[i];
  return result;
}


// Vector-Vector subtraction (pointwise difference)
template<typename T1, typename T2>
auto operator-(const Vector<T1>& l,
	       const Vector<T2>& r)
{
  int size_l = l.size();
  int size_r = r.size();
  if (size_l != size_r) throw "Vectors have different size!";
  
  Vector<typename std::common_type<T1,T2>::type> result(size_l,0);
  for (auto i = 0; i < size_l; ++i)
    result.set(i, l.get(i) - r.get(i));
  //v.get[i] = v1.get[i] + v2.get[i];
  return result;
}

// Vector-Scalar addition
template<typename T1, typename T2>
auto operator+(const Vector<T1>& vec,
	       const        T2 & scalar)
{
  int size = vec.size();
  Vector<typename std::common_type<T1,T2>::type> result(size,0);
  
  for (auto i = 0; i < size; ++i)
    result.set(i, vec.get(i) + scalar);
  return result;
}

// Scalar-Vector addition
template<typename T1, typename T2>
auto operator+(const        T1   scalar,
	       const Vector<T2> &vec)
{
  return vec + scalar;
}
// Vector-Scalar subtraction
template<typename T1, typename T2>
auto operator-(const Vector<T1>& vec,
	       const        T2 & scalar)
{
  int size = vec.size();
  Vector<typename std::common_type<T1,T2>::type> result(size,0);
  
  for (auto i = 0; i < size; ++i)
    result.set(i, vec.get(i) - scalar);
  return result;
}

// Scalar-Vector subtraction
template<typename T1, typename T2>
auto operator-(const        T1   scalar,
	       const Vector<T2> &vec)
{
  return vec - scalar;
}

// Vector-Scalar multiplication
template<typename T1, typename T2>
auto operator*(const Vector<T1> &vec,
	       const        T2   scalar)
{
  Vector<typename std::common_type<T1,T2>::type> result(vec);

  for (auto i = 0; i < result.size(); ++i){
    result.set(i,result.get(i) * scalar);
  }

  return result;
}

// Scalar-Vector multiplication
template<typename T1, typename T2>
auto operator*(const T1 scalar, const Vector<T2> &vec)
{
  return vec * scalar;
}

// Vector-Vector multiplication (dot product)
template<typename T1, typename T2>
auto dot(const Vector<T1>& l,
	 const Vector<T2>& r)
{
  int size_l = l.size();
  int size_r = r.size();
  if (size_l != size_r) throw "Vectors have different size!";

  typename std::common_type<T1,T2>::type result = 0;
  for (auto i = 0; i < size_l; i++)
    result += l.get(i) * r.get(i);
  
  return result;
}



/////////////////////////////////////////////////////////////////////
//
//
//   TEMPLATE CLASS MATRIX
//
//
/////////////////////////////////////////////////////////////////////


//template<typename K = std::array<int, 2>, typename T = double>
template<typename T>
class Matrix{

public:
  const int length;
  const int width;
  std::map<std::array<int, 2>, T> grid;

  
  //
  //  CONSTRUCTORS
  // --------------
  //

  // Default constructor
  Matrix():
    length(0),
    width(0)
  {
    // std::cout << "Default constructor" << std::endl;
  }

  
  // Constructor (sets length, width)
  Matrix(const int length, const int width):
    length(length),
    width(width)
  {
    // std::cout << "Length/Width constructor" << std::endl;
  }

  //
  //  OPERATORS
  // --------------
  //

  // Returns the reference of the value of the key
  auto& operator[](const std::array<int,2> key)
  {
    return grid[key];
  }
  
  // Returns the const reference of the value of the key
  // Does not allow change
  const auto& operator[](const std::array<int,2> key) const
  {
    return grid[key];
  }

  //
  //  FUNCTIONS
  // -----------
  //

  // Naive Matrix-Vector multiplication
  // Attempts to multiply ALL elements
  // Vector<T> matvec_naive(const Vector<T> vec) const
  // {
  //   int size = vec.size();

  //   if (this->width != size) throw "The width of the matrix is different than the length of the vector!";

  //   Vector<T> result(this->length);
  //   T sum;
  //   for (auto i = 0; i < this->length; ++i){
  //     sum = 0;
  //     for (auto j = 0; j < this->width; ++j){
  // 	sum = sum + vec.get(j) * this->grid.at({i,j});
  //     }
  //     result.set(i,sum);
  //   }
  //   return result;
  // }



  // Sparse Matrix-Vector multiplication.
  // Based on the COO format. The coordinatesare taken from
  // the key of the map that is behind the matrix.
  Vector<T> matvec2(const Vector<T> vec) const
  {

    int size = vec.size();
    
    if (this->width != size) throw "The width of the matrix is different than the length of the vector!";
    
    Vector<T> result(this->length, 0);

    for (auto it = grid.begin(); it != grid.end(); ++it){
      // std::cout << "Elem: m[" << it->first[0] << ", "  << it->first[1] << "] = " << it->second << std::endl;
      // std::cout << "Mult: v[" << it->first[1] << "] = " << vec.get(it->first[1]) << std::endl;
      // std::cout << "Into: r[" << it->first[0] << "] = " << result.get(it->first[0])  << std::endl;
      // std::cout << "--------------------------------" << std::endl;

      result[it->first[0]] = result[it->first[0]] + it->second * vec[it->first[1]];
    }

    return result;
  }


  // Prints the non-zero elements of the matrix on the default output
  void print() const {
    for (auto it = grid.begin(); it != grid.end(); ++it){
      std::cout << "m["   << it->first[0]
		<< ", "   << it->first[1]
		<< "] = " << it->second
		<< std::endl;
    }
  }

};


/////////////////////////////////////////////////////////////////////
//
//
//   CONJUGATE GRADIENT SOLVER TEMPLATE
//
//
/////////////////////////////////////////////////////////////////////


template<typename T>
int cg(const Matrix<T> &A, const Vector<T> &b, Vector<T> &x, T tol, int maxiter)
{

  // Search Direction vector
  Vector<T> p;
  
  try
    {
      p = b - A.matvec2(x);
    }
  catch (const char* msg)
    {
      std::cerr << msg << std::endl;
      // If either the matrix-vector multiplication or the vector-vector addition fails,
      // there is no point in the program "staying alive".
      // If they both succeed, then there is no point in catching throws again.
      exit(1);
    }

  // Residual vector
  Vector<T> r(p);

  T alpha_k;
  T beta_k;
  T tol2 = tol * tol;

  // Scalar variable for storing the Euclidean norm for r_{k}
  // Calculate and save the first square Euclidean norm
  T norm_rkk = dot(r, r);

  // Scalar variable for storing the Euclidean norm for r_{k+1}
  T norm_rk1;

  // Vector variable for storing the Matrix-Vector multiplication
  // that occurs twice in the cg loop
  Vector<T> ap_k;


  auto k = 0;
  for (k = 0; k < maxiter; ++k) {

    // Matrix-Vector multimplication A*p
    ap_k = A.matvec2(p);

    // Division of scalars (square norm over a dot product)
    alpha_k = norm_rkk / dot(ap_k, p);
    
    // Solution vector update
    x = x + (alpha_k *  p);

    // Residual vector update
    r = r - (alpha_k * ap_k);

    // Square Norm of the residual
    norm_rk1 = dot(r, r);


    // STOPPING CRITERION
    if (norm_rk1 < tol2) {
      break;
    }

    // Division of square norms (scalar)
    beta_k = norm_rk1/ norm_rkk;

    // Search direction vector update
    p = r + (beta_k * p);

    // Update square norm (no need to be recalculated)
    norm_rkk = norm_rk1;
  }


  // Return the number of iterations needed until convergence
  // If the maximum number of iterations is exceeded, return -1
  if (k < maxiter)
    {
      return k;
    }
  else
    {
      return -1;
    }
}

/////////////////////////////////////////////////////////////////////
//
//
//   CLASS HEAD 1D
//
//
/////////////////////////////////////////////////////////////////////


//template<typename T>
class Heat1D
{

private:

public:
  // ATTRIBUTES

  const double alpha;
  const int m;
  const double dt;

  double dx;

  // Discretization matrix
  Matrix<double> M;
  // Dimension of the discretization matrix
  int m_dim;

  const double tol = TOL;
  const int maxiter = MAXIT;

  
  // Constructors

  //---------------

  // Default constructor
  Heat1D():
    alpha(0),
    m(0),
    dt(0)
  {}

  // Constructor
  Heat1D(const double alpha , const int m, const double dt):
    alpha(alpha),
    m(m),
    dt(dt),
    M(m,m)
  {
    m_dim = m;

    // Distance between nodes
    dx = 1.0 / ( m + 1.0 );

    // Coefficient for the discretization matrix
    const double c  = (alpha * dt) / ( dx * dx );

    // Fill the matrix with the appropriate elements
    int j;
    for (int i = 0; i < m_dim; ++i){
      
      // Diagonal elements
      M[{i,i}] = 1 + 2 * c;

      // Second diagonal elements
      j = i + 1;
      if ( j % m != 0){
     	M[{i,j}] = - c;
     	M[{j,i}] = - c;
      }
    }


    /// TEST  ////
    
    // Vector<double> vec(m_dim, 1);
    // std::cout << M.matvec2(vec) << std::endl;
    
    // M.print();
  }


  // Computes the exact solution
  // We return by value because we declare the vector in the function.
  // We expect a compiler optimazation in the "receiving" end.
  Vector<double> exact(const double t) const{

    Vector<double> x(m_dim);

    const double expo = exp( - PI * PI * alpha * t);

    for (auto i = 0; i < x.size(); ++i){
      x[i] = expo * sin(PI*(i+1)*dx);
    }

    return x;
  }


  // Solves the initial boundary value problem (ibvp)
  // We return by value because we declare the vector in the function.
  // We expect a compiler optimazation in the "receiving" end.
  Vector<double> solve(double t_end) const{

    Vector<double> x = std::move(exact(0.));

    int limit = round(t_end / this->dt);                      // avoid??

    for (auto i = 0; i < limit; ++i){                         // make better??
      // std::cout << "t = " << i*dt << std::endl;
      // std::cout << x << std::endl;
      if( cg(M, x, x, tol, maxiter) == -1 ){
    	std::cerr << "ERROR! Failed to converge!" << std::endl;
	break;
      }
    }

    return x;
  }

  // Prints the solution (the vector provided by the user
  // based on the grid location of each node
  void print_solution(const Vector<double> x) const{

    for (auto i = 0; i < m_dim; ++i){
      std::cout << i << ' ' << x[i] << std::endl;
    }
  }
  
  // Saves the solution (the vector provided by the user
  // based on the grid location of each node
  int save_solution(const Vector<double>& x, const std::string & name) const{

    std::string filename = "solution_1d_" + name + ".dat";
    std::ofstream outfile(filename);
    if (outfile.is_open()){
      for (auto i = 0; i < m_dim; ++i){
	outfile  << i << ' ' << x[i] << '\n';
      }
      outfile.close();
    }
    else{
      std::cerr << "Unable to open file" << std::endl;
    }
    return 0;
  }
  

};



/////////////////////////////////////////////////////////////////////
//
//
//   CLASS HEAD 2D
//
//
/////////////////////////////////////////////////////////////////////


//template<typename T>
class Heat2D
{

private:

public:

  // ATTRIBUTES

  const double alpha;
  const int m;
  const double dt;

  double dx;
  
  // Discretization matrix
  Matrix<double> M;
  // Dimension of the discretization matrix
  int m_dim;

  const double tol = TOL;
  const int maxiter = MAXIT;


  // Constructors

  //---------------

  //Default constructor
  Heat2D():
    alpha(0),
    m(0),
    dt(0)
  {}

  // Constructor
  Heat2D(const double alpha , const int m, const double dt):
    alpha(alpha),
    m(m),
    dt(dt),
    M(m*m,m*m)
  {

    m_dim = m*m;

    // Distance between nodes
    dx = 1.0 / ( m + 1.0 );

    // Coefficient for the discretization matrix
    const double c  = (alpha * dt) / ( dx * dx );

    // Limit for the third diagonal
    const int limit = m_dim;

    // Fill the matrix with the appropriate elements
    int j;
    for (int i = 0; i < m_dim; ++i){

      // Diagonal elements
      M[{i,i}] = 1 + 2 * c * 2;

      // Second diagonal elements
      j = i + 1;
      if ( j % m != 0){
     	M[{i,j}] = - c;
     	M[{j,i}] = - c;
      }

      // Third diagonal elements
      j = i + m;
      if ( j < limit ){
	M[{i,j}] = - c;
	M[{j,i}] = - c;
      }
    }


    /// TEST  ////

    // Vector<double> vec(m_dim, 1);
    // std::cout << M.matvec2(vec) << std::endl;
    
    // M.print();
  }

  // Computes the exact solution
  // We return by value because we declare the vector in the function.
  // We expect a compiler optimazation in the "receiving" end.
  Vector<double> exact(const double t) const{

    Vector<double> x(m_dim);

    const double expo = exp( - 2 * PI * PI * alpha * t);

    const double pidx = PI*dx;

    for (auto k = 0; k < m_dim; ++k){
      x[k] = expo;
      x[k] *= ( sin(pidx*(k / m + 1)) *
		sin(pidx*(k % m + 1)) );
    }

    return x;
  }

  // Solves the initial boundary value problem (ibvp)
  // We return by value because we declare the vector in the function.
  // We expect a compiler optimazation in the "receiving" end.
  Vector<double> solve(double t_end) const{

    // Create solution vector and initialize
    Vector<double> x = std::move(exact(0.));

    const int limit = round(t_end / this->dt);                      // avoid??
    
    for (auto i = 0; i < limit; ++i){                                // make better??
      // std::cout << "t = " << i * dt << std::endl;
      // std::cout << x << std::endl;
      if( cg(M, x, x, tol, maxiter) == -1 ){
    	std::cerr << "ERROR! Failed to converge!" << std::endl;
	break;
      }
    }

    return x;
  }

  // Prints the solution (the vector provided by the user
  // based on the grid location of each node
  void print_solution(const Vector<double>& x) const{

    for (auto k = 0; k < m_dim; ++k){
      std::cout << k / m << ' '  << k % m << ' ' << x[k] << '\n';
    }
  }

  // Saves the solution (the vector provided by the user
  // based on the grid location of each node
  int save_solution(const Vector<double>& x, const std::string & name) const{

    std::string filename = "solution_2d_" + name + ".dat";
    std::ofstream outfile(filename);
    if (outfile.is_open()){
      for (auto k = 0; k < m_dim; ++k){
	outfile << k / m << ' '  << k % m << ' ' << x[k] << '\n';
      }
      outfile.close();
    }
    else{
      std::cerr << "Unable to open file" << std::endl;
    }
    return 0;
  }
  
};


/////////////////////////////////////////////////////////////////////
//
//
//   CALCULATE INTEGER POW(base,power)
//
//
/////////////////////////////////////////////////////////////////////

// It would be more efficient if we turned it into a template
// with 2 specializations for the cases: power == 0 and power == 1
inline int ipow(const int base, const int power){

  if (power == 0) return 1;
  if (power == 1) return base;
  return base * ipow(base, power - 1);

}


/////////////////////////////////////////////////////////////////////
//
//
//   CLASS HEAD n-DIAMENSIONAL
//
//
/////////////////////////////////////////////////////////////////////

template<int n>
class HeatnD
{

private:

public:

  // ATTRIBUTES

  const double alpha;
  const int m;
  const double dt;

  double dx;
  
  // Discretization matrix
  Matrix<double> M;
  // Dimension of the discretization matrix
  int m_dim;

  const double tol = TOL;
  const int maxiter = MAXIT;


  //
  //  CONSTRUCTORS
  // --------------
  //

  // Default constructor
  HeatnD():
    alpha(0),
    m(0),
    dt(0)
  {}

  // Constructor
  HeatnD(const double alpha , const int m, const double dt):
    alpha(alpha),
    m(m),
    dt(dt),
    M(ipow(m,n),ipow(m,n))
  {

    // Calculate the size of the solution vector
    m_dim = ipow(m,n);

    // Distance between nodes
    dx = 1.0 / ( m + 1.0 );
    
    // Coefficient for the discretization matrix
    const double c  = (alpha * dt) / ( dx * dx );

    // Counter for the dimentions
    int d;

    // Vector with m raised to the power of the index
    // Reduces processing power at the cost of memory
    Vector<int> m_pow(n+1);
    m_pow[0] = 1;
    for (d = 0; d < n; ++d){
      m_pow[d+1] = m_pow[d] * m;
    }

    // Vector with limits for the off-diagonal elements
    Vector<int> m_lim(n);
    for (d = 0; d < n; ++d){
      m_lim[d] = m_pow[d+1];
    }

    // Flag for the update of the limits for the off-diagonal elements
    bool flag = false;
    
    // Fill the matrix with the appropriate elements
    int j;
    for (int i = 0; i < m_dim; ++i){

      // Diagonal elements
      M[{i,i}] = 1 + 2 * c * n;

      // Second diagonal elements
      j = i + 1;
      if ( j % m != 0){
     	M[{i,j}] = - c;
     	M[{j,i}] = - c;
      }
      else {
	flag = true;
      }

      // Limit update of the off-diagonal emelents for the third (and more) dimensions
      if ( flag && i % m == 0){
	for (d = 1; d < n; ++d){
	  if ( i % m_pow[d+1] == 0){
	    m_lim[d] = m_lim[d] + m_pow[d+1];
	  }
	}
	flag = false;
      }

      // Off-diagonal emelents for the third (and more) dimensions
      for (d = 1; d < n; ++d){
	j = i + m_pow[d];
	if ( j < m_lim[d] ){
	  M[{i,j}] = - c;
	  M[{j,i}] = - c;
	}
      }

    }

    /// TEST  ////

    // Vector<double> vec(m_dim, 1);
    // std::cout << M.matvec2(vec) << std::endl;
    
    // M.print();
  }


  //
  //  FUNCTIONS
  // -----------
  //

  // Returns a vector with the coordinates of node p on the grid
  // based on its lexicographical ordering (the input)
  // It would be more efficient if we could turn it into a recursive
  // function and then into a template with specializations
  const Vector<int> pos(int p) const
  {

    Vector<int> vec(n);

    int temp;
    for (int k = n - 1; k > 0; k--){
      temp   = ipow(m,k);
      vec[k] = p / temp;
      p      = p % temp;
    }

    vec[0] = p;

    return vec;
  }

  // Calculates recursively the non-exponential part of the
  // initialization/exact solution vector
  // It would be more efficient if we turned it into a template
  // with a specialization for d == 0.
  inline const double calc_exact_term(const Vector<int> & p, const int d, const double c) const{
    if (d == 0)
      return sin(c*(p[d]+1));
    else
      return sin(c*(p[d]+1)) * calc_exact_term(p,d-1,c);
  }


  // Computes the exact solution
  // We return by value because we declare the vector in the function.
  // We expect a compiler optimazation in the "receiving" end.
  Vector<double> exact(const double t) const{

    // Solution vector
    Vector<double> x(m_dim);
    
    // Vector for the coordinates of a node
    Vector<int> p;

    const int d = n - 1;
    const double expo = exp( - n * PI * PI * alpha * t);
    const double pidx = PI*dx;

    for (auto i = 0; i < x.size(); ++i){
      x[i] = expo * calc_exact_term(pos((int)i),d, pidx);
    }

    return x;
  }

  // Solves the initial boundary value problem (ibvp)
  // We return by value because we declare the vector in the function.
  // We expect a compiler optimazation in the "receiving" end.
  Vector<double> solve(const double t_end) const{

    // Create solution vector and initialize
    Vector<double> x = std::move(exact(0.));

    const int limit = round(t_end / this->dt);                          // avoid??

    for (auto i = 0; i < limit; ++i){                                   // make better??
      // std::cout << "t = " << i * dt << std::endl;
      // std::cout << x << std::endl;
      if( cg(M, x, x, tol, maxiter) == -1 ){
    	std::cerr << "ERROR! Failed to converge!" << std::endl;
	break;
      }
    }

    return x;
  }
  // Prints the solution (the vector provided by the user
  // based on the grid location of each node
  void print_solution(const Vector<double>& x) const{

    Vector<int> p;
    for (auto i = 0; i < m_dim; ++i){
      p = pos((int)i);
      for (auto j = 0; j < p.size(); ++j){
	std::cout << p[j] << ' ';
      }
      std::cout << x[i] << '\n';
    }
  }

  // Saves the solution (the vector provided by the user
  // based on the grid location of each node
  int save_solution(const Vector<double>& x, const std::string & name) const{

    std::string filename = "solution_n" + std::to_string(n) + "d_" + name + ".dat";
    std::ofstream outfile(filename);
    if (outfile.is_open()){
      Vector<int> p;
      for (auto i = 0; i < m_dim; ++i){
	p = pos((int)i);
	for (auto j = 0; j < p.size(); ++j){
	  outfile << p[j] << ' ';
	}
	outfile << x[i] << '\n';
      }
      outfile.close();
    }
    else{
      std::cerr << "Unable to open file" << std::endl;
    }
    return 0;
  }
  

};
  
/////////////////////////////////////////////////////////////////////
//
//
//   MAIN
//
//
/////////////////////////////////////////////////////////////////////



// The global main function that is the designated start of the program
int main(){

  // Input parameters of the problem
  
  const double alpha = 0.3125;
  const double dt    = 0.1;
  const int m = 9;

  const double t_end = 1;


  //
  // TEST HEAT 1D
  //

  // std::cout << "RUNNING HEAT 1D" << std::endl;

  // Heat1D ibvp1d(alpha, m, dt);

  // std::cout << "Running solve 1d..." << std::endl;
  // Vector<double> sol_anal_1d = ibvp1d.solve(t_end);
  // // std::cout << sol_anal_1d << std::endl;
  // // ibvp1d.print_solution(sol_anal_1d);
  // ibvp1d.save_solution(sol_anal_1d, "analytical");
  
  // std::cout << "Running exact 1d..." << std::endl;
  // Vector<double> sol_exac_1d = ibvp1d.exact(t_end);
  // // std::cout << sol_exac_1d << std::endl;
  // // ibvp1d.print_solution(sol_exac_1d);
  // ibvp1d.save_solution(sol_exac_1d, "exact");
  
  // std::cout << "Comparing with tol = " << ibvp1d.tol << std::endl;
  // std::cout << "Element-wise: " <<  sol_anal_1d.compare(sol_exac_1d, ibvp1d.tol ) << std::endl;
  // std::cout << "2 norm:       " <<  sol_anal_1d.diff_norm(sol_exac_1d, ibvp1d.tol ) << std::endl;
  
  //
  // TEST HEAT ND FOR N == 1
  //

  // std::cout << "RUNNING HEAT ND, N == 1" << std::endl;
  
  // HeatnD<1> ibvpnd1(alpha, m, dt);

  // std::cout << "Running solve nd..." << std::endl;
  // Vector<double> sol_anal_nd1 = ibvpnd1.solve(t_end);
  // // std::cout << sol_anal_nd1 << std::endl;
  // // ibvpnd1.print_solution(sol_anal_nd1);
  // ibvpnd1.save_solution(sol_anal_nd1, "analytical");
  
  // std::cout << "Running exact nd..." << std::endl;
  // Vector<double> sol_exac_nd1 = ibvpnd1.exact(t_end);
  // // std::cout << sol_exac_nd1 << std::endl;
  // // ibvpnd1.print_solution(sol_exac_nd1);
  // ibvpnd1.save_solution(sol_exac_nd1, "exact");
  
  // std::cout << "Comparing with tol = " << ibvpnd1.tol << std::endl;
  // std::cout << "Element-wise: " <<  sol_anal_nd1.compare(sol_exac_nd1, ibvpnd1.tol ) << std::endl;
  // std::cout << "2 norm:       " <<  sol_anal_nd1.diff_norm(sol_exac_nd1, ibvpnd1.tol ) << std::endl;
  
  //
  // TEST HEAT 2D
  //

  std::cout << "RUNNING HEAT 2D" << std::endl;

  Heat2D ibvp2d(alpha, m, dt);

  std::cout << "Running solve 2d..." << std::endl;
  // std::cout << "init x: " << ibvp1d.x << std::endl;
  Vector<double> sol_anal_2d = ibvp2d.solve(t_end);
  // std::cout << sol_anal_2d << std::endl;
  // ibvp2d.print_solution(sol_anal_2d);
  ibvp2d.save_solution(sol_anal_2d, "analytical");
  
  std::cout << "Running exact 2d..." << std::endl;
  Vector<double> sol_exac_2d = ibvp2d.exact(t_end);
  // std::cout << sol_exac_2d << std::endl;
  // ibvp2d.print_solution(sol_exac_2d);
  ibvp2d.save_solution(sol_exac_2d, "exact");

  std::cout << "Comparing with tol = " << ibvp2d.tol << std::endl;
  std::cout << "Element-wise: " <<  sol_anal_2d.compare(sol_exac_2d, ibvp2d.tol ) << std::endl;
  std::cout << "2 norm:       " <<  sol_anal_2d.diff_norm(sol_exac_2d, ibvp2d.tol ) << std::endl;

  
  
  //
  // TEST HEAT ND FOR N == 2
  //
  
  std::cout << "RUNNING HEAT ND, N == 2" << std::endl;
  
  HeatnD<2> ibvpnd2(alpha, m, dt);

  std::cout << "Running solve nd..." << std::endl;
  Vector<double> sol_anal_nd2 = ibvpnd2.solve(t_end);
  // std::cout << sol_anal_nd2 << std::endl;
  // ibvpnd2.print_solution(sol_anal_nd2);
  ibvpnd2.save_solution(sol_anal_nd2, "analytical");
  
  std::cout << "Running exact nd..." << std::endl;
  Vector<double> sol_exac_nd2 = ibvpnd2.exact(t_end);
  // std::cout << sol_exac_nd2 << std::endl;
  // ibvpnd2.print_solution(sol_exac_nd2);
  ibvpnd2.save_solution(sol_exac_nd2, "exact");
  
  std::cout << "Comparing with tol = " << ibvpnd2.tol << std::endl;
  std::cout << "Element-wise: " <<  sol_anal_nd2.compare(sol_exac_nd2, ibvpnd2.tol ) << std::endl;
  std::cout << "2 norm:       " <<  sol_anal_nd2.diff_norm(sol_exac_nd2, ibvpnd2.tol ) << std::endl;


  //
  // TEST HEAT ND FOR N == 3
  //

  // std::cout << "RUNNING HEAT ND, N == 3" << std::endl;

  // HeatnD<3> ibvpnd3(alpha, m, dt);

  // std::cout << "Running solve nd..." << std::endl;
  // Vector<double> sol_anal_nd3 = ibvpnd3.solve(t_end);
  // // std::cout << sol_anal_nd3 << std::endl;
  // // ibvpnd3.print_solution(sol_anal_nd3);
  // ibvpnd3.save_solution(sol_anal_nd3, "analytical");
  
  // std::cout << "Running exact nd..." << std::endl;
  // Vector<double> sol_exac_nd3 = ibvpnd3.exact(t_end);
  // // std::cout << sol_exac_nd3 << std::endl;
  // // ibvpnd3.print_solution(sol_exac_nd3);
  // ibvpnd3.save_solution(sol_exac_nd3, "exact");
  
  // std::cout << "Comparing with tol = " << ibvpnd3.tol << std::endl;
  // std::cout << "Element-wise: " <<  sol_anal_nd3.compare(sol_exac_nd3, ibvpnd3.tol ) << std::endl;
  // std::cout << "2 norm:       " <<  sol_anal_nd3.diff_norm(sol_exac_nd3, ibvpnd3.tol ) << std::endl;


  return 0;
}

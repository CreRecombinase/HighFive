//
// Created by nwknoblauch on 12/30/17.
//

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>
#include <typeinfo>
#include <vector>


#include <memory>


#ifdef USE_BLOSC

#include <H5Ppublic.h>
#include <highfive/H5Filter.hpp>
#include "blosc_filter.h"

#endif


#include <cstdio>
#include <sys/stat.h>
#include <highfive/H5File.hpp>
#include <highfive/H5Utility.hpp>
#include <highfive/EigenUtils.hpp>

#define BOOST_TEST_MODULE HighFiveEigenTest

#include <boost/mpl/list.hpp>
#include <boost/test/unit_test.hpp>


inline bool file_exists(const std::string &name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

void remove_if_exists(const std::string &filen) {
    if (file_exists(filen)) {
        remove(filen.c_str());
    }
}

using namespace HighFive;
typedef boost::mpl::list<float, double> floating_numerics_test_types;
typedef boost::mpl::list<int, unsigned int, long, unsigned long, unsigned char,
        char, float, double, long long, unsigned long long>
        numerical_test_types;
typedef boost::mpl::list<int, unsigned int, long, unsigned long, unsigned char,
        char, float, double, std::string>
        dataset_test_types;

template<typename T, typename Func>
void generate2D(T *table, size_t x, size_t y, Func &func) {
    for (size_t i = 0; i < x; i++) {
        for (size_t j = 0; j < y; j++) {
            table[i][j] = func();
        }
    }
}

template<typename T, typename Func>
void generate2D(std::vector<std::vector<T> > &vec, size_t x, size_t y,
                Func &func) {
    vec.resize(x);
    for (size_t i = 0; i < x; i++) {
        vec[i].resize(y);
        for (size_t j = 0; j < y; j++) {
            vec[i][j] = func();
        }
    }
}

template<typename T>
struct ContentGenerate {
    ContentGenerate(T init_val = T(0), T inc_val = T(1) + T(1) / T(10))
            : _init(init_val), _inc(inc_val) {}

    T operator()() {
        T ret = _init;
        _init += _inc;
        return ret;
    }

    T _init, _inc;
};

template<>
struct ContentGenerate<char> {
    ContentGenerate() : _init('a') {}

    char operator()() {
        char ret = _init;
        if (++_init >= ('a' + 26))
            _init = 'a';
        return ret;
    }

    char _init;
};

template<>
struct ContentGenerate<std::string> {
    ContentGenerate() {}

    std::string operator()() {
        ContentGenerate<char> gen;
        std::string random_string;
        const int size_string = std::rand() % 254;
        random_string.resize(size_string);
        std::generate(random_string.begin(), random_string.end(), gen);
        return random_string;
    }
};

#ifdef USE_BLOSC

template<typename T>
void eigen_matrix_compression_Test() {

    typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Matrix;

    std::ostringstream filename;
    filename << "h5_rw_matrix_eigen_compression_rm_" << typeid(T).name() << "_test.h5";
    remove_if_exists(filename.str());

    const size_t size_x = 10, size_y = 10;
    const std::string DATASET_NAME("dset");

    Matrix mat(size_x, size_y);

    ContentGenerate<T> generator;
    for (Eigen::Index i = 0; i < mat.rows(); ++i) {
        for (Eigen::Index j = 0; j < mat.cols(); ++j) {
            mat(i, j) = generator();
        }
    }
    // Create a new file using the default property lists.
    File file(filename.str(), File::ReadWrite | File::Create | File::Truncate);

    char *version, *date;
    auto r = register_blosc(&version, &date);
    free(version);
    free(date);
    std::vector<size_t> cshape{5, 5};

    Filter filter(cshape, FILTER_BLOSC, r);
    // Create a dataset with double precision floating points
    DataSet dataset = file.createDataSet(DATASET_NAME, DataSpace::From(mat), AtomicType<T>(), filter.getId());

    dataset.write(mat);

    // read it back
    Matrix result;

    dataset.read(result);
    BOOST_CHECK(mat.isApprox(result));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(eigen_matrix_compression, T, numerical_test_types) {

    eigen_matrix_compression_Test<T>();
}

#endif


template<typename T>
void eigen_matrix_Test() {

    typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Matrix;

    std::ostringstream filename;
    filename << "h5_rw_matrix_eigen_rm_" << typeid(T).name() << "_test.h5";
    remove_if_exists(filename.str());
    const size_t size_x = 9, size_y = 10;
    const std::string DATASET_NAME("dset");

    Matrix mat(size_x, size_y);

    ContentGenerate<T> generator;
    for (Eigen::Index i = 0; i < mat.rows(); ++i) {
        for (Eigen::Index j = 0; j < mat.cols(); ++j) {
            mat(i, j) = generator();
        }
    }
    // Create a new file using the default property lists.

    File file(filename.str(), File::ReadWrite | File::Create | File::Truncate);
    auto tgrp = file.getGroup("/");


    DataSet dataset = tgrp.createDataSet<T>(DATASET_NAME, DataSpace::From(mat));
    dataset.write(mat);

    // read it back
    Matrix result;

    dataset.read(result);
    BOOST_CHECK(mat.isApprox(result));

}

BOOST_AUTO_TEST_CASE_TEMPLATE(eigen_matrix, T, numerical_test_types) {

    eigen_matrix_Test<T>();
}


template<typename T>
void R_eigen_matrix_w_Test() {

    typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Matrix;

    std::ostringstream filename;
    filename << "h5_rw_matrix_R_w_eigen_rm_" << typeid(T).name() << "_test.h5";

    remove_if_exists(filename.str());
    const size_t size_x = 9, size_y = 10;
    const std::string DATASET_NAME("dset");

    Matrix mat(size_x, size_y);

    ContentGenerate<T> generator;
    for (Eigen::Index i = 0; i < mat.rows(); ++i) {
        for (Eigen::Index j = 0; j < mat.cols(); ++j) {
            mat(i, j) = generator();
        }
    }
    // Create a new file using the default property lists.
    write_mat_h5(filename.str(), "/", "dset", mat);

    File file(filename.str(), File::ReadOnly);
    auto tgrp = file.getGroup("/");

    Matrix result;
    DataSet dataset = tgrp.getDataSet("dset");
    dataset.read(result);

    BOOST_CHECK(mat.isApprox(result));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(R_eigen_w_matrix, T, numerical_test_types) {

    R_eigen_matrix_w_Test<T>();
}


template<typename T>
void R_eigen_matrix_w_t_Test() {

    typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Matrix;

    std::ostringstream filename;
    filename << "h5_rw_matrix_R_w_t_eigen_rm_" << typeid(T).name() << "_test.h5";

    remove_if_exists(filename.str());
    const size_t size_x = 9, size_y = 10;
    const std::string DATASET_NAME("dset");

    Matrix mat(size_x, size_y);

    ContentGenerate<T> generator;
    for (Eigen::Index i = 0; i < mat.rows(); ++i) {
        for (Eigen::Index j = 0; j < mat.cols(); ++j) {
            mat(i, j) = generator();
        }
    }
    // Create a new file using the default property lists.
    write_mat_h5(filename.str(), "/", "dset", mat, true);

    Matrix result;

    read_mat_h5(filename.str(), "/", "dset", result);


    BOOST_CHECK(mat.isApprox(result));

}

BOOST_AUTO_TEST_CASE_TEMPLATE(R_eigen_w_t_matrix, T, numerical_test_types) {

    R_eigen_matrix_w_t_Test<T>();
}


template<typename T>
void R_eigen_matrix_r_t_c_Test() {

    typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Matrix;

    std::ostringstream filename;
    filename << "h5_rw_matrix_R_r_t_c_eigen_rm_" << typeid(T).name() << "_test.h5";

    remove_if_exists(filename.str());
    const size_t size_x = 9, size_y = 10;
    const std::string DATASET_NAME("dset");

    Matrix mat(size_x, size_y);

    ContentGenerate<T> generator;
    for (Eigen::Index i = 0; i < mat.rows(); ++i) {
        for (Eigen::Index j = 0; j < mat.cols(); ++j) {
            mat(i, j) = generator();
        }
    }
    // Create a new file using the default property lists.

    write_mat_h5(filename.str(), "/", "dset", mat, true);

    Matrix result;
    Matrix check_res = mat.block(1, 2, 4, 5);

    read_mat_h5(filename.str(), "/", "dset", result, {1, 2}, {4, 5});

    BOOST_CHECK(result.isApprox(check_res));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(R_eigen_r_t_c_matrix, T, numerical_test_types) {

    R_eigen_matrix_r_t_c_Test<T>();
}


template<typename T>
void R_eigen_matrix_w_t_c_Test() {

    typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Matrix;

    std::ostringstream filename;
    filename << "h5_rw_matrix_R_w_t_c_eigen_rm_" << typeid(T).name() << "_test.h5";

    remove_if_exists(filename.str());
    const size_t size_x = 9, size_y = 10;
    const std::string DATASET_NAME("dset");

    Matrix mat(size_x, size_y);

    ContentGenerate<T> generator;
    for (Eigen::Index i = 0; i < mat.rows(); ++i) {
        for (Eigen::Index j = 0; j < mat.cols(); ++j) {
            mat(i, j) = generator();
        }
    }
    Matrix check_res = mat.block(1, 2, 4, 5);
    // Create a new file using the default property lists.
    {
        File file(filename.str(), File::Create | File::ReadWrite);
        auto dataset = file.createDataSet("dset", DataSpace::From(mat, true), AtomicType<T>(), H5P_DEFAULT, true);
        dataset.selectEigen({1, 2}, {4, 5}, {}).write(check_res);

    }
    //write_mat_h5(filename.str(), "/", "dset", mat, true);

    Matrix result;
    //Matrix check_res = mat.block(1, 2, 4, 5);

    read_mat_h5(filename.str(), "/", "dset", result, {1, 2}, {4, 5});

    BOOST_CHECK(result.isApprox(check_res));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(R_eigen_w_t_c_matrix, T, numerical_test_types) {

    R_eigen_matrix_w_t_c_Test<T>();
}


template<typename T>
void R_eigen_matrix_r_Test() {

    typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Matrix;

    std::ostringstream filename;
    filename << "h5_rw_matrix_R_r_eigen_rm_" << typeid(T).name() << "_test.h5";

    const size_t size_x = 10, size_y = 10;
    const std::string DATASET_NAME("dset");

    Matrix mat(size_x, size_y);

    ContentGenerate<T> generator;
    for (Eigen::Index i = 0; i < mat.rows(); ++i) {
        for (Eigen::Index j = 0; j < mat.cols(); ++j) {
            mat(i, j) = generator();
        }
    }
    // Create a new file using the default property lists.
    {
        File file(filename.str(), File::ReadWrite | File::Create | File::Truncate);
        auto tgrp = file.getGroup("/");


        DataSet dataset = tgrp.createDataSet<T>(DATASET_NAME, DataSpace::From(mat));
        dataset.write(mat);
    }

    // read it back
    Matrix result;
    read_mat_h5<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>(filename.str(), "/", "dset", result);

    BOOST_CHECK(result.isApprox(mat));

}

BOOST_AUTO_TEST_CASE_TEMPLATE(R_eigen_r_matrix, T, numerical_test_types) {

    R_eigen_matrix_r_Test<T>();
}


template<typename T>
void eigen_matrix_map_Test() {

    typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef typename Eigen::Map<Matrix> MapMat;

    std::ostringstream filename;
    filename << "h5_rw_matrix_map_eigen_rm_" << typeid(T).name() << "_test.h5";

    const size_t size_x = 10, size_y = 10;
    const std::string DATASET_NAME("dset");

    Matrix mat(size_x, size_y);

    ContentGenerate<T> generator;
    for (Eigen::Index i = 0; i < mat.rows(); ++i) {
        for (Eigen::Index j = 0; j < mat.cols(); ++j) {
            mat(i, j) = generator();
        }
    }
    // Create a new file using the default property lists.
    File file(filename.str(), File::ReadWrite | File::Create | File::Truncate);


    DataSet dataset = file.createDataSet<T>(DATASET_NAME, DataSpace::From(mat));
    dataset.write(mat);

    // read it back
    Matrix result(size_x, size_y);
    MapMat map_res(result.data(), result.rows(), result.cols());

    dataset.read(map_res);
    BOOST_CHECK(result.isApprox(mat));

}

BOOST_AUTO_TEST_CASE_TEMPLATE(eigen_matrix_map, T, numerical_test_types) {

    eigen_matrix_map_Test<T>();
}


template<typename T>
void eigen_matrix_map_rm_Test() {

    typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix_rm;
    typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Matrix_cm;
    typedef typename Eigen::Map<Matrix_rm> MapMat_rm;
    typedef typename Eigen::Map<Matrix_cm> MapMat_cm;

    std::ostringstream filename;
    filename << "h5_rw_matrix_map_eigen_rm_" << typeid(T).name() << "_test.h5";

    remove_if_exists(filename.str());
    const size_t size_x = 9, size_y = 10;
    const std::string DATASET_NAME("dset");

    Matrix_rm mat(size_x, size_y);

    ContentGenerate<T> generator;
    for (Eigen::Index i = 0; i < mat.rows(); ++i) {
        for (Eigen::Index j = 0; j < mat.cols(); ++j) {
            mat(i, j) = generator();
        }
    }
    // Create a new file using the default property lists.
    File file(filename.str(), File::ReadWrite | File::Create | File::Truncate);


    DataSet dataset = file.createDataSet<T>(DATASET_NAME, DataSpace::From(mat));
    dataset.write(mat);
    Matrix_rm tmat = mat.transpose();

    // read it back
    Matrix_cm result(size_y, size_x);
    MapMat_rm map_res(result.data(), size_x, size_y);

    dataset.read(map_res);

    BOOST_CHECK(result.isApprox(tmat));

}

BOOST_AUTO_TEST_CASE_TEMPLATE(eigen_matrix_map_rm, T, numerical_test_types) {

    eigen_matrix_map_rm_Test<T>();
}


//Ensure that the type conversion performed by the library is equivalent to the one performed by Eigen
template<typename T>
void eigen_fd_matrix_Test() {

    typedef typename std::conditional<std::is_same<T, double>::value, float, double>::type OT;
    typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Matrix_T;
    typedef typename Eigen::Matrix<OT, Eigen::Dynamic, Eigen::Dynamic> Matrix_OT;


    std::ostringstream filename;
    filename << "h5_rw_matrix_eigen_rm_" << typeid(T).name() << "_test.h5";

    const size_t size_x = 10, size_y = 10;
    const std::string DATASET_NAME("dset");

    Matrix_T mat(size_x, size_y);

    ContentGenerate<T> generator;
    for (Eigen::Index i = 0; i < mat.rows(); ++i) {
        for (Eigen::Index j = 0; j < mat.cols(); ++j) {
            mat(i, j) = generator();
        }
    }
    Matrix_OT check_mat(mat.template cast<OT>());
    // Create a new file using the default property lists.
    File file(filename.str(), File::ReadWrite | File::Create | File::Truncate);


    DataSet dataset = file.createDataSet<T>(DATASET_NAME, DataSpace::From(mat));
    dataset.write(mat);

    // read it back
    Matrix_OT result;

    dataset.read(result);

    BOOST_CHECK(check_mat.isApprox(result));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(eigen_fd_matrix, T, floating_numerics_test_types) {

    eigen_fd_matrix_Test<T>();
}


template<typename T>
void eigen_cm_rm_matrix_Test() {

    typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Matrix_cm;
    typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix_rm;

    std::ostringstream filename;
    filename << "h5_rw_matrix_cm_rm_" << typeid(T).name() << "_test.h5";

    remove_if_exists(filename.str());
    const size_t size_x = 9, size_y = 10;
    const std::string DATASET_NAME("dset");

    Matrix_rm mat(size_x, size_y);

    ContentGenerate<T> generator;
    for (Eigen::Index i = 0; i < mat.rows(); ++i) {
        for (Eigen::Index j = 0; j < mat.cols(); ++j) {
            mat(i, j) = generator();
        }
    }

    // Create a new file using the default property lists.
    File file(filename.str(), File::ReadWrite | File::Create | File::Truncate);

    DataSet dataset = file.createDataSet<T>(DATASET_NAME, DataSpace::From(mat));

    dataset.write(mat);

    // read it back
    Matrix_cm result;

    dataset.read(result);
    BOOST_CHECK(mat.isApprox(result));

}

BOOST_AUTO_TEST_CASE_TEMPLATE(eigen_cm_rm_matrix, T, numerical_test_types) {

    eigen_cm_rm_matrix_Test<T>();
}


template<typename T>
void eigen_rm_cm_matrix_Test() {

    typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Matrix_cm;
    typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix_rm;

    std::ostringstream filename;
    filename << "h5_rw_matrix_cm_rm_" << typeid(T).name() << "_test.h5";

    remove_if_exists(filename.str());
    const size_t size_x = 9, size_y = 10;
    const std::string DATASET_NAME("dset");

    Matrix_cm mat(size_x, size_y);

    ContentGenerate<T> generator;

    for (Eigen::Index i = 0; i < mat.rows(); ++i) {
        for (Eigen::Index j = 0; j < mat.cols(); ++j) {
            mat(i, j) = generator();
        }
    }

    // Create a new file using the default property lists.
    File file(filename.str(), File::ReadWrite | File::Create | File::Truncate);

    DataSet dataset = file.createDataSet<T>(DATASET_NAME, DataSpace::From(mat));

    dataset.write(mat);

    // read it back
    Matrix_rm result;

    dataset.read(result);
    BOOST_CHECK(mat.isApprox(result));

}

BOOST_AUTO_TEST_CASE_TEMPLATE(eigen_rm_cm_matrix, T, numerical_test_types) {

    eigen_rm_cm_matrix_Test<T>();
}


template<typename T>
void eigen_rm_slice_matrix_Test() {

    typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Matrix_cm;
    typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix_rm;
    std::ostringstream filename;
    filename << "h5_rw_matrix_cm_rm_" << typeid(T).name() << "_test.h5";
    remove_if_exists(filename.str());
    const size_t size_x = 10, size_y = 10;
    const std::string DATASET_NAME("dset");

    Matrix_rm mat(size_x, size_y);

    ContentGenerate<T> generator;
    for (Eigen::Index i = 0; i < mat.rows(); ++i) {
        for (Eigen::Index j = 0; j < mat.cols(); ++j) {
            mat(i, j) = generator();
        }
    }

    // Create a new file using the default property lists.
    File file(filename.str(), File::ReadWrite | File::Create | File::Truncate);

    DataSet dataset = file.createDataSet<T>(DATASET_NAME, DataSpace::From(mat));

    Matrix_rm tmat = mat.block(0, 0, 5, 5);
    dataset.select({0, 0}, {5, 5}).write(tmat);

    tmat = mat.block(0, 5, 5, 5);
    dataset.select({0, 5}, {5, 5}).write(tmat);

    tmat = mat.block(5, 0, 5, 5);
    dataset.select({5, 0}, {5, 5}).write(tmat);

    tmat = mat.block(5, 5, 5, 5);
    dataset.select({5, 5}, {5, 5}).write(tmat);

    // read it back
    Matrix_rm result;

    dataset.read(result);
    BOOST_CHECK(mat.isApprox(result));

}

BOOST_AUTO_TEST_CASE_TEMPLATE(eigen_rm_slice_matrix, T, numerical_test_types) {

    eigen_rm_slice_matrix_Test<T>();
}

template<typename T>
void eigen_cm_slice_matrix_Test() {

    typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Matrix_cm;
    typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix_rm;

    std::ostringstream filename;
    filename << "h5_rw_matrix_cm_rm_" << typeid(T).name() << "_test.h5";
    remove_if_exists(filename.str());
    const size_t size_x = 10, size_y = 10;
    const std::string DATASET_NAME("dset");

    Matrix_cm mat(size_x, size_y);

    ContentGenerate<T> generator;
    for (Eigen::Index i = 0; i < mat.rows(); ++i) {
        for (Eigen::Index j = 0; j < mat.cols(); ++j) {
            mat(i, j) = generator();
        }
    }

    // Create a new file using the default property lists.
    File file(filename.str(), File::ReadWrite | File::Create | File::Truncate);

    DataSet dataset = file.createDataSet<T>(DATASET_NAME, DataSpace({size_x, size_y}));

    Matrix_cm tmat = mat.block(0, 0, 5, 5);
    dataset.select({0, 0}, {5, 5}).write(tmat);

    tmat = mat.block(0, 5, 5, 5);
    dataset.select({0, 5}, {5, 5}).write(tmat);

    tmat = mat.block(5, 0, 5, 5);
    dataset.select({5, 0}, {5, 5}).write(tmat);

    tmat = mat.block(5, 5, 5, 5);
    dataset.select({5, 5}, {5, 5}).write(tmat);

    // read it back
    Matrix_cm result;

    dataset.read(result);
    BOOST_CHECK(mat.isApprox(result));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(eigen_cm_slice_matrix, T, numerical_test_types) {

    eigen_cm_slice_matrix_Test<T>();
}




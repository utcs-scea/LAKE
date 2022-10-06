#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/ndarrayobject.h>
#include "py_wrapper.h"
#include <stdio.h>
#include <unistd.h>
#include <limits.h>

PyObject *kleio_pDict = NULL;

int kleio_load_model(const char *filepath) {
    wchar_t** _argv = PyMem_Malloc(sizeof(wchar_t*)*1);
    wchar_t* arg = Py_DecodeLocale("test", NULL);
    _argv[0] = arg;

    Py_Initialize();
    PySys_SetArgv(1, _argv);
    _import_array();

    PyObject* sysPath = PySys_GetObject("path");
    PyList_Append(sysPath, PyUnicode_FromString(__FILE__));
    PyObject *moduleString = PyUnicode_FromString("run_cluster_lstm");
    PyObject *PyPredict = PyImport_Import(moduleString);
    if (!PyPredict) {
        PyErr_Print();
        printf("ERROR in pModule\n");
        return -1;
    } 

    kleio_pDict = PyModule_GetDict(PyPredict);
    PyObject *loadModelFunc = PyDict_GetItem(kleio_pDict, PyUnicode_FromString("kleio_load_model"));
    
    if (loadModelFunc == NULL) {
        printf("load python func failed\n");
        return -1;
    }

    PyObject *pyResult = PyObject_CallFunction(loadModelFunc, "s", filepath);
    if (PyLong_Check(pyResult) != 1) {
        PyErr_Print();
        printf("load_model return error val\n");
    }
    int ret = (int) PyLong_AsLong(pyResult);
    return ret;
}

PyObject *makearray(int *array, size_t size) {
    // npy_intp dim = size;
    // PyObject *new_array = PyArray_SimpleNewFromData(1, &dim, NPY_INT, (void *)array);
    // printf("created array with %d\n", size);
    // return new_array;
    PyObject* python_list = PyList_New(size);
    Py_INCREF(python_list);
    int* ar = (int*) array;
    for (int i = 0; i < size; ++i) {
        PyObject* python_int = Py_BuildValue("i", ar[i]);
        PyList_SetItem(python_list, i, python_int);
    }
    return python_list;
}

double kleio_inference(const void *syscalls, unsigned int num_syscall) {
    PyObject *standardInferenceFunc = PyDict_GetItem(kleio_pDict, PyUnicode_FromString("kleio_inference"));
    if (standardInferenceFunc == NULL) {
        printf("load inference python func failed\n");
        return -1;
    }

    /* Marshall args */
    PyObject *pArgs = PyTuple_New(2);
    PyTuple_SetItem(pArgs, 0, makearray((int *)syscalls, num_syscall));
    PyTuple_SetItem(pArgs, 1, PyLong_FromUnsignedLong((unsigned long)num_syscall));
    PyObject *pyResult = PyObject_CallObject(standardInferenceFunc, pArgs);
    double elapsed = PyFloat_AsDouble(pyResult);
    if (elapsed == -1.0) {
        printf("kleio_inference return error val\n");
        return -1;
    }

    return elapsed;
}

void kleio_force_gc(void) {
    PyObject *func = PyDict_GetItem(kleio_pDict, PyUnicode_FromString("kleio_force_gc"));
    PyObject *pyResult = PyObject_CallObject(func, 0);
    if (!pyResult) {
        printf("dogc failed\n");
    }
}

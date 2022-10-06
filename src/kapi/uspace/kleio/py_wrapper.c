#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/ndarrayobject.h>
#include "py_wrapper.h"
#include <stdio.h>
#include <unistd.h>
#include <limits.h>

PyObject *kleio_pDict = NULL;

int kleio_load_model(const char *filepath) {
    printf("loading kleio at %s\n", filepath);

    wchar_t** _argv = PyMem_Malloc(sizeof(wchar_t*)*1);
    wchar_t* arg = Py_DecodeLocale("test", NULL);
    _argv[0] = arg;

    Py_Initialize();
    PySys_SetArgv(1, _argv);
    _import_array();

    PyObject* sysPath = PySys_GetObject("path");
    char *libpath = "/disk/hfingler/HACK/kava/worker/lstm_tf/lstm_tf_wrapper/coeus-sim-master";

    PyList_Append(sysPath, PyUnicode_FromString(libpath));

    PyObject *moduleString = PyUnicode_FromString("run_cluster_lstm");
    PyObject *PyPredict = PyImport_Import(moduleString);
    if (!PyPredict) {
        PyErr_Print();
        printf("ERROR in pModule\n");
        return -1;
    } else {
        kleio_pDict = PyModule_GetDict(PyPredict);
        PyObject *loadModelFunc = PyDict_GetItem(kleio_pDict, PyUnicode_FromString("kleio_load_model"));
        
        if (loadModelFunc != NULL) {
            PyObject *pyResult = PyObject_CallFunction(loadModelFunc, "s", filepath);
            if (PyLong_Check(pyResult) != 1) {
                PyErr_Print();
                printf("load_model return error val\n");
            }
            int ret = (int) PyLong_AsLong(pyResult);
            return ret;
        } else {
            printf("load python func failed\n");
            return -1;
        }
    }
    //Py_Finalize();
    return 0;
}

PyObject *makearray(int *array, size_t size) {
    npy_intp dim = size;
    PyObject *new_array = PyArray_SimpleNewFromData(1, &dim, NPY_INT, (void *)array);
    return new_array;
}

void kleio_close_ctx(void) {
    PyObject *func = PyDict_GetItem(kleio_pDict, PyUnicode_FromString("print_stats"));
    PyObject *pyResult = PyObject_CallObject(func, 0);
    if (!pyResult) {
        printf("close_ctx failed\n");
    }
    fflush(stdout);
}

int kleio_inference(const void *syscalls, unsigned int num_syscall, unsigned int sliding_window) {
    PyObject *standardInferenceFunc = PyDict_GetItem(kleio_pDict, PyUnicode_FromString("kleio_inference"));

    if (standardInferenceFunc != NULL) {
        /* Marshall args */
        PyObject *pArgs = PyTuple_New(3);
        PyTuple_SetItem(pArgs, 0, makearray((int *)syscalls, num_syscall));
        PyTuple_SetItem(pArgs, 1, PyLong_FromUnsignedLong((unsigned long)num_syscall));
        PyTuple_SetItem(pArgs, 2, PyLong_FromUnsignedLong((unsigned long)sliding_window));
        PyObject *pyResult = PyObject_CallObject(standardInferenceFunc, pArgs);
        if (PyLong_Check(pyResult) != 1) {
            printf("kleio_inference return error val");
            return -1;
        }
        int ret = (int) PyLong_AsLong(pyResult);
        return ret;
    } else {
        printf("load inference python func failed\n");
        return -1;
    }
    return -1;
}


void dogc(void) {
    if (!kleio_pDict) {
        PyObject* sysPath = PySys_GetObject("path");
        char *libpath = "/disk/hfingler/HACK/kava/worker/lstm_tf/lstm_tf_wrapper/coeus-sim-master";
        PyList_Append(sysPath, PyUnicode_FromString(libpath));
        PyObject *moduleString = PyUnicode_FromString("run_cluster_lstm");
        PyObject *PyPredict = PyImport_Import(moduleString);
        kleio_pDict = PyModule_GetDict(PyPredict);
    }

    PyObject *func = PyDict_GetItem(kleio_pDict, PyUnicode_FromString("dogc"));
    PyObject *pyResult = PyObject_CallObject(func, 0);
    if (!pyResult) {
        printf("dogc failed\n");
    }
    fflush(stdout);
}

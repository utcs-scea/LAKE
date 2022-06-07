#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/ndarrayobject.h>
#include "c_wrapper.h"
#include <stdio.h>
#include <unistd.h>
#include <limits.h>

PyObject *pDict = NULL;

PyObject *makearray(int *array, size_t size) {
    npy_intp dim = size;
    /* int dim[1]; */
    /* dim[0] = size; */
    PyObject *new_array = PyArray_SimpleNewFromData(1, &dim, NPY_INT, (void *)array);
    return new_array;
}


int load_model(const char *filepath) {
    /* char *filepath = "/home/edwardhu/kava/worker/lstm_tf/lstm_tf_wrapper/"; */
    printf("%s\n", filepath);
    //Py_SetProgramName("test");
    //PyRun_SimpleString("import sys\n"
    //                 "sys.argv = []");

    wchar_t** _argv = PyMem_Malloc(sizeof(wchar_t*)*1);
    //for (int i=0; i<argc; i++) {
    wchar_t* arg = Py_DecodeLocale("test", NULL);
    _argv[0] = arg;
    //}

    Py_Initialize();
    printf("a\n");
    PySys_SetArgv(1, _argv);

    import_array();

    printf("1\n");


    PyObject* sysPath = PySys_GetObject("path");
    /* char cwd[PATH_MAX]; */
    /* if (getcwd(cwd, sizeof(cwd)) != NULL) { */
    /*     printf("Current working dir: %s\n", cwd); */
    /* } else { */
    /*     perror("getcwd() error"); */
    /*     return -1; */
    /* } */

    //char *libpath = "/home/edwardhu/kava/worker/lstm_tf/lstm_tf_wrapper/";
    char *libpath = "/home/hfingler/hf-HACK/kava/worker/lstm_tf/lstm_tf_wrapper";
    printf("1.5\n");
    PyList_Append(sysPath, PyUnicode_FromString(libpath));

    printf("2\n");

    /* PyRun_SimpleString("import numpy as np"); */
    /* PyRun_SimpleString("import os"); */
    /* PyRun_SimpleString("import tensorflow as tf"); */
    /* PyRun_SimpleString("from tensorflow import keras"); */

    //PyObject *PyFileDir = PyString_FromString((char *)filepath);
    /* PyObject *PyFileDir = PyUnicode_FromString((char *)filepath); */
    /* PyObject *PyFileDir = Py_BuildValue("s", filepath); */

    PyObject *moduleString = PyUnicode_FromString("predict");
    PyObject *PyPredict = PyImport_Import(moduleString);
    if (!PyPredict)
    {
        PyErr_Print();
        printf("ERROR in pModule\n");
        return -1;
    } else {
        pDict = PyModule_GetDict(PyPredict);
        PyObject *loadModelFunc = PyDict_GetItem(pDict, PyUnicode_FromString("load_model"));
        
        if (loadModelFunc != NULL) {
            /* PyObject *pyResult = PyObject_CallObject(loadModelFunc, PyFileDir); */
            PyObject *pyResult = PyObject_CallFunction(loadModelFunc, "s", filepath);
            if (PyLong_Check(pyResult) != 1) {
                printf("load_model return error val");
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

int standard_inference(const void *syscalls, unsigned int num_syscall, unsigned int sliding_window) {
    PyObject *standardInferenceFunc = PyDict_GetItem(pDict, PyUnicode_FromString("standard_inference"));
    if (standardInferenceFunc != NULL) {
        /* Marshall args */
        PyObject *pArgs = PyTuple_New(3);

        PyTuple_SetItem(pArgs, 0, makearray((int *)syscalls, num_syscall));
        PyTuple_SetItem(pArgs, 1, PyLong_FromUnsignedLong((unsigned long)num_syscall));
        PyTuple_SetItem(pArgs, 2, PyLong_FromUnsignedLong((unsigned long)sliding_window));
        PyObject *pyResult = PyObject_CallObject(standardInferenceFunc, pArgs);
        if (PyLong_Check(pyResult) != 1) {
            printf("standard_inference return error val");
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

void close_ctx(void) {
    // gc
    PyObject *closeCtxFunc = PyDict_GetItem(pDict, PyUnicode_FromString("close_ctx"));
    PyObject_CallFunction(closeCtxFunc, NULL);
    Py_Finalize();
}
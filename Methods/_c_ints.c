#include <Python.h>
#include <numpy/npy_no_deprecated_api.h>
#include <numpy/arrayobject.h>
#include "c_ints/shellpair_quantities.h"
#include "c_ints/two_electron_bound.h"
#include "c_ints/two_electron_fundamentals.h"
#include "c_ints/two_electron_scattering.h"
#include "c_ints/two_electron_vrr.h"
#include "c_ints/two_electron_contract.h"
#include "c_ints/two_electron_hrr.h"
#include "c_ints/one_electron_fundamentals.h"
#include "c_ints/one_electron_vrr.h"
#include "c_ints/one_electron_hrr.h"
#include "c_ints/one_electron_kinetic.h"
#include "c_ints/one_electron_contract.h"

/*---------------------------------------------------------------------*/
/*                           General setup                             */
/*---------------------------------------------------------------------*/

/* Docstrings for this module and wrappers within it                   */
/* To add a new function, copy the last 2 lines and modify accordingly */
/*                                                                     */
static char module_docstring[] = 
   "This module provides an interface from python to C integral routines";
static char shellpair_docstring[] = 
   "Calculate shell-pair quantities, takes basis data and zeroed arrays for shell-pair data as input";
static char bound_docstring[] = 
   "Calculate Schwarz bound/s for current shell-quartet from existing Coulomb integrals";
static char fundamentals_2e_docstring[] = 
   "Calculate fundamental integrals for all primitives in current shell-quartet";
static char vrr_2e_docstring[] = 
   "Apply vertical recurrence relations to form uncontracted [m0|n0] integrals";
static char contract_2e_docstring[] = 
   "Contract [m0|n0] integrals to form (m0|n0) integrals";
static char hrr_2e_docstring[] = 
   "Apply horizontal recurrence relations to form contracted (mn|ls) integrals";
static char fundamentals_1e_docstring[] = 
   "Calculate fundamental nuclear attraction integrals for current shell-pair";
static char vrr_1e_docstring[] = 
   "Apply vertical recurrence relations to form uncontracted [m|0] nuclear attraction and overlap integrals";
static char hrr_1e_docstring[] = 
   "Apply horizontal recurrence relations to form uncontracted [m|n] nuclear attraction and overlap integrals";
static char kinetic_1e_docstring[] = 
   "Compute uncontracted kinetic [m|n] integrals from [m||n+2], [m||n] and [m||n-2] using derivative formula";
static char contract_1e_docstring[] = 
   "Contract [m|n] integrals to form (m|n) integrals";

/* Declare wrapper functions                                           */
/*                                                                     */
static PyObject *c_ints_shellpair_quantities(PyObject *self, PyObject *args);
static PyObject *c_ints_two_electron_bound(PyObject *self, PyObject *args);
static PyObject *c_ints_two_electron_fundamentals(PyObject *self, PyObject *args);
static PyObject *c_ints_two_electron_vrr(PyObject *self, PyObject *args);
static PyObject *c_ints_two_electron_contract(PyObject *self, PyObject *args);
static PyObject *c_ints_two_electron_hrr(PyObject *self, PyObject *args);
static PyObject *c_ints_one_electron_fundamentals(PyObject *self, PyObject *args);
static PyObject *c_ints_one_electron_vrr(PyObject *self, PyObject *args);
static PyObject *c_ints_one_electron_hrr(PyObject *self, PyObject *args);
static PyObject *c_ints_one_electron_kinetic(PyObject *self, PyObject *args);
static PyObject *c_ints_one_electron_contract(PyObject *self, PyObject *args);

/* Set up python methods to access wrapper functions from python       */
/* First entry is name of wrapped function as seen by python, second   */
/* is name of wrapper function                                         */
/* To add a new method, copy the second line and modify accordingly    */ 
/*                                                                     */
static PyMethodDef module_methods[] = {
   {"shellpair_quantities", c_ints_shellpair_quantities, METH_VARARGS, shellpair_docstring},
   {"two_electron_bound", c_ints_two_electron_bound, METH_VARARGS, bound_docstring},
   {"two_electron_fundamentals", c_ints_two_electron_fundamentals, METH_VARARGS, fundamentals_2e_docstring},
   {"two_electron_vrr", c_ints_two_electron_vrr, METH_VARARGS, vrr_2e_docstring},
   {"two_electron_contract", c_ints_two_electron_contract, METH_VARARGS, contract_2e_docstring},
   {"two_electron_hrr", c_ints_two_electron_hrr, METH_VARARGS, hrr_2e_docstring},
   {"one_electron_fundamentals", c_ints_one_electron_fundamentals, METH_VARARGS, fundamentals_1e_docstring},
   {"one_electron_vrr", c_ints_one_electron_vrr, METH_VARARGS, vrr_1e_docstring},
   {"one_electron_hrr", c_ints_one_electron_hrr, METH_VARARGS, hrr_1e_docstring},
   {"one_electron_kinetic", c_ints_one_electron_kinetic, METH_VARARGS, kinetic_1e_docstring},
   {"one_electron_contract", c_ints_one_electron_contract, METH_VARARGS, contract_1e_docstring},
   {NULL, NULL, 0, NULL} /* Sentinel, all methods now defined */
};

/* Initialize module once and for all                                  */
/*                                                                     */
#if PY_MAJOR_VERSION >= 3
// Python3.x version
PyMODINIT_FUNC PyInit__c_ints(void) {

   /* Set up module */
   PyObject *module;
   static struct PyModuleDef moduledef = { PyModuleDef_HEAD_INIT, "_c_ints",
      module_docstring, -1, module_methods, NULL, NULL, NULL, NULL
   }; 
   module = PyModule_Create(&moduledef);
   if (!module) return NULL;

   /* Load 'numpy' functionality */
   import_array();

   return module;
}
#else
// Python 2.x version
PyMODINIT_FUNC init_c_ints(void) {
   (void) Py_InitModule3("_c_ints", module_methods, module_docstring);
   import_array();
}
#endif

/*---------------------------------------------------------------------*/
/*                     Subroutine-specific wrappers                    */
/*---------------------------------------------------------------------*/

static PyObject *c_ints_shellpair_quantities(PyObject *self, PyObject *args) {

   /* Declare types for objects received from python */
   int n_alpha, n_beta;
   PyObject *alpha_exponents_obj, *A_obj, *beta_exponents_obj, *B_obj;
   PyObject *sigmas_obj, *overlaps_obj, *centres_obj;

   /* C pointers to the arrays contained within those objects */
   double *alpha_exponents, *A, *beta_exponents, *B;
   double **sigmas, **overlaps, ***centres;

   /* Output variables that capture properties of input arrays */
   npy_intp dims[3];
   int typenum = NPY_DOUBLE;
   PyArray_Descr *descr;
   descr = PyArray_DescrFromType(typenum);

   /* Parse the input tuple */
   if (!PyArg_ParseTuple(args, "OOOOOiOOi", &sigmas_obj, &overlaps_obj, &centres_obj,
                         &alpha_exponents_obj, &A_obj, &n_alpha, &beta_exponents_obj, &B_obj, &n_beta))
       {
          PyErr_SetString(PyExc_TypeError, "Error parsing objects passed to C");
          return NULL;
       }
 
   /* Extract pointers to memory locations directly, without copying data to C */
   if (PyArray_AsCArray(&sigmas_obj, (void **)&sigmas, dims, 2, descr) < 0) return NULL;
   if (PyArray_AsCArray(&overlaps_obj, (void **)&overlaps, dims, 2, descr) < 0) return NULL;
   if (PyArray_AsCArray(&alpha_exponents_obj, (void *)&alpha_exponents, dims, 1, descr) < 0) return NULL;
   if (PyArray_AsCArray(&A_obj, (void *)&A, dims, 1, descr) < 0) return NULL;
   if (PyArray_AsCArray(&beta_exponents_obj, (void *)&beta_exponents, dims, 1, descr) < 0) return NULL;
   if (PyArray_AsCArray(&B_obj, (void *)&B, dims, 1, descr) < 0) return NULL;
   if (PyArray_AsCArray(&centres_obj, (void ***)&centres, dims, 3, descr) < 0) return NULL;

   /* Call the external C function to assign values to sigmas, overlaps, centres arrays in-place */
   shellpair_quantities(sigmas, overlaps, centres, alpha_exponents, A, n_alpha, beta_exponents, B, n_beta); 

   /* No clean up required, arrays and pointers borrowed from python */
   /* Return none */
   return Py_BuildValue(""); 

}

/*---------------------------------------------------------------------*/
static PyObject *c_ints_two_electron_bound(PyObject *self, PyObject *args) {

   /* Declare types for objects received from python */
   int nla, nlb, nlc, nld;
   PyObject *C_P_obj, *C_Q_obj, *bounds_obj;

   /* C pointers to the arrays contained within those objects */
   double **C_P, **C_Q, *bounds;
    
   /* Output variables that capture properties of input arrays */
   npy_intp dims[3];
   int typenum = NPY_DOUBLE;
   PyArray_Descr *descr;
   descr = PyArray_DescrFromType(typenum);

   /* Parse the input tuple */
   if (!PyArg_ParseTuple(args, "OOOiiii", &bounds_obj, &C_P_obj, &C_Q_obj,
                         &nla, &nlb, &nlc, &nld))
       {
          PyErr_SetString(PyExc_TypeError, "Error parsing objects passed to C");
          return NULL;
       }
   
   /* Extract pointers to memory locations directly, without copying data to C */
   if (PyArray_AsCArray(&C_P_obj, (void **)&C_P, dims, 2, descr) < 0) return NULL;
   if (PyArray_AsCArray(&C_Q_obj, (void **)&C_Q, dims, 2, descr) < 0) return NULL;
   if (PyArray_AsCArray(&bounds_obj, (void *)&bounds, dims, 1, descr) < 0) return NULL;

   /* Call the external C function to compute values of bounds in-place, using computed indexing */
   two_electron_bound(bounds, C_P, C_Q, nla, nlb, nlc, nld);

   /* Return none */
   return Py_BuildValue(""); 

}

/*---------------------------------------------------------------------*/
static PyObject *c_ints_two_electron_fundamentals(PyObject *self, PyObject *args) {

   /* Declare types for objects received from python */
   int na, nb, nc, nd, m, ints_type;
   double grid_value;
   PyObject *sigma_P_obj, *U_P_obj, *P_obj;
   PyObject *sigma_Q_obj, *U_Q_obj, *Q_obj;
   PyObject *R_obj, *fundamentals_obj;

   /* C pointers to the arrays contained within those objects */
   double **sigma_P, **U_P, ***P;
   double **sigma_Q, **U_Q, ***Q;
   double ***R, ***fundamentals;

   /* Output variables that capture properties of input arrays */
   npy_intp dims[3];
   int typenum = NPY_DOUBLE;
   PyArray_Descr *descr;
   descr = PyArray_DescrFromType(typenum);

   /* Parse the input tuple */
   if (!PyArg_ParseTuple(args, "OOOOOOOOiiiiiid", &fundamentals_obj, 
                         &sigma_P_obj, &U_P_obj, &P_obj,
                         &sigma_Q_obj, &U_Q_obj, &Q_obj, &R_obj,
                         &na, &nb, &nc, &nd, &m, &ints_type, &grid_value))
       {
          PyErr_SetString(PyExc_TypeError, "Error parsing objects passed to C");
          return NULL;
       }
   
   /* Extract pointers to memory locations directly, without copying data to C */
   if (PyArray_AsCArray(&fundamentals_obj, (void ***)&fundamentals, dims, 3, descr) < 0) return NULL;
   if (PyArray_AsCArray(&sigma_P_obj, (void **)&sigma_P, dims, 2, descr) < 0) return NULL;
   if (PyArray_AsCArray(&U_P_obj, (void **)&U_P, dims, 2, descr) < 0) return NULL;
   if (PyArray_AsCArray(&P_obj, (void ***)&P, dims, 3, descr) < 0) return NULL;
   if (PyArray_AsCArray(&sigma_Q_obj, (void **)&sigma_Q, dims, 2, descr) < 0) return NULL;
   if (PyArray_AsCArray(&U_Q_obj, (void **)&U_Q, dims, 2, descr) < 0) return NULL;
   if (PyArray_AsCArray(&Q_obj, (void ***)&Q, dims, 3, descr) < 0) return NULL;
   if (PyArray_AsCArray(&R_obj, (void ***)&R, dims, 3, descr) < 0) return NULL;

   /* Call the external C function to compute values of fundamental integrals in-place, using computed indexing */
   if (ints_type == 1) {
      two_electron_scattering(fundamentals, sigma_P, U_P, P, sigma_Q, U_Q, Q, R, grid_value, na, nb, nc, nd, m);
   } else {
      two_electron_fundamentals(fundamentals, sigma_P, U_P, P, sigma_Q, U_Q, Q, R, na, nb, nc, nd, m);
   }

   /* Return none */
   return Py_BuildValue(""); 

}

/*---------------------------------------------------------------------*/

static PyObject *c_ints_two_electron_vrr(PyObject *self, PyObject *args) {

   /* Declare types for objects received from python */
   int na, nb, nc, nd, lbra, lket, kappa_index, n_base_classes;
   PyObject *target_ints_obj, *base_ints_obj;
   PyObject *zeta_obj, *eta_obj, *kappa_obj, *Rx_obj, *R_obj; 

   /* Declare types for sub-objects within base_ints object */
   PyObject *base_int0_obj, *base_int1_obj, *base_int2_obj, *base_int3_obj, *base_int4_obj;

   /* C pointers to the arrays contained within those objects */
   double **target_ints, **base_int0, **base_int1, **base_int2, **base_int3, **base_int4;
   double **zeta, **eta, *kappa, *Rx, ***R; 

   /* Output variables that capture properties of input arrays */
   npy_intp dims[3];
   int typenum = NPY_DOUBLE;
   PyArray_Descr *descr;
   descr = PyArray_DescrFromType(typenum);

   /* Parse the input tuple */
   if (!PyArg_ParseTuple(args, "OOOOOOOiiiiiiii", &target_ints_obj, &base_ints_obj,
                         &zeta_obj, &eta_obj, &kappa_obj, &Rx_obj, &R_obj, 
                         &n_base_classes, &na, &nb, &nc, &nd, &lbra, &lket, &kappa_index))
      {
         PyErr_SetString(PyExc_TypeError, "Error parsing objects passed to C");
         return NULL;
      }
   
   /* Extract pointers to memory locations directly, without copying data to C */
   if (PyArray_AsCArray(&target_ints_obj, (void **)&target_ints, dims, 2, descr) < 0) return NULL;
   if (PyArray_AsCArray(&zeta_obj, (void **)&zeta, dims, 2, descr) < 0) return NULL;
   if (PyArray_AsCArray(&eta_obj, (void **)&eta, dims, 2, descr) < 0) return NULL;
   if (PyArray_AsCArray(&kappa_obj, (void *)&kappa, dims, 1, descr) < 0) return NULL;
   if (PyArray_AsCArray(&Rx_obj, (void *)&Rx, dims, 1, descr) < 0) return NULL;
   if (PyArray_AsCArray(&R_obj, (void ***)&R, dims, 3, descr) < 0) return NULL;

   /* Assign pointers to ints within base_ints list, dummy to base_int0 if not supplied */
   base_int0_obj = PyList_GetItem(base_ints_obj, (Py_ssize_t) 0);
   base_int1_obj = PyList_GetItem(base_ints_obj, (Py_ssize_t) 1);
   if (n_base_classes >= 3) { base_int2_obj = PyList_GetItem(base_ints_obj, (Py_ssize_t) 2); } 
   if (n_base_classes >= 4) { base_int3_obj = PyList_GetItem(base_ints_obj, (Py_ssize_t) 3); } 
   if (n_base_classes == 5) { base_int4_obj = PyList_GetItem(base_ints_obj, (Py_ssize_t) 4); } 

   if (PyArray_AsCArray(&base_int0_obj, (void **)&base_int0, dims, 2, descr) < 0) return NULL;
   if (PyArray_AsCArray(&base_int1_obj, (void **)&base_int1, dims, 2, descr) < 0) return NULL;
   if (n_base_classes == 2) {
     if (PyArray_AsCArray(&base_int0_obj, (void **)&base_int2, dims, 2, descr) < 0) return NULL;
     if (PyArray_AsCArray(&base_int0_obj, (void **)&base_int3, dims, 2, descr) < 0) return NULL;
     if (PyArray_AsCArray(&base_int0_obj, (void **)&base_int4, dims, 2, descr) < 0) return NULL;
   }
   if (n_base_classes == 3) {
     if (PyArray_AsCArray(&base_int0_obj, (void **)&base_int2, dims, 2, descr) < 0) return NULL;
     if (PyArray_AsCArray(&base_int0_obj, (void **)&base_int3, dims, 2, descr) < 0) return NULL;
     if (PyArray_AsCArray(&base_int2_obj, (void **)&base_int4, dims, 2, descr) < 0) return NULL;
   }
   if (n_base_classes == 4) {
     if (PyArray_AsCArray(&base_int2_obj, (void **)&base_int2, dims, 2, descr) < 0) return NULL;
     if (PyArray_AsCArray(&base_int3_obj, (void **)&base_int3, dims, 2, descr) < 0) return NULL;
     if (PyArray_AsCArray(&base_int0_obj, (void **)&base_int4, dims, 2, descr) < 0) return NULL;
   }
   if (n_base_classes == 5) {
     if (PyArray_AsCArray(&base_int2_obj, (void **)&base_int2, dims, 2, descr) < 0) return NULL;
     if (PyArray_AsCArray(&base_int3_obj, (void **)&base_int3, dims, 2, descr) < 0) return NULL;
     if (PyArray_AsCArray(&base_int4_obj, (void **)&base_int4, dims, 2, descr) < 0) return NULL;
   }

   /* Call external C function to compute values of [m0|n0] integrals in-place, 
      using computed indexing within bra and ket */

   two_electron_vrr(target_ints, base_int0, base_int1, base_int2, base_int3, base_int4, 
                    zeta, eta, kappa, Rx, R, na, nb, nc, nd, lbra, lket, kappa_index); 

   /* Return none */
   return Py_BuildValue(""); 

}

/*---------------------------------------------------------------------*/

static PyObject *c_ints_two_electron_contract(PyObject *self, PyObject *args) {

   /* Declare types for objects received from python */
   int na, nb, nc, nd, lbra, lket;
   PyObject *contracted_ints_obj, *primitive_ints_obj;
   PyObject *cc_bra_obj, *cc_ket_obj; 

   /* C pointers to the arrays contained within those objects */
   double **primitive_ints, **contracted_ints;
   double **cc_bra, **cc_ket;

   /* Output variables that capture properties of input arrays */
   npy_intp dims[3];
   int typenum = NPY_DOUBLE;
   PyArray_Descr *descr;
   descr = PyArray_DescrFromType(typenum);

   /* Parse the input tuple */
   if (!PyArg_ParseTuple(args, "OOOOiiiiii", &contracted_ints_obj, &primitive_ints_obj, 
                         &cc_bra_obj, &cc_ket_obj, &na, &nb, &nc, &nd, &lbra, &lket))
      {
         PyErr_SetString(PyExc_TypeError, "Error parsing objects passed to C");
         return NULL;
      }
   
   /* Extract pointers to memory locations directly, without copying data to C */
   if (PyArray_AsCArray(&contracted_ints_obj, (void **)&contracted_ints, dims, 2, descr) < 0) return NULL;
   if (PyArray_AsCArray(&primitive_ints_obj, (void **)&primitive_ints, dims, 2, descr) < 0) return NULL;
   if (PyArray_AsCArray(&cc_bra_obj, (void *)&cc_bra, dims, 2, descr) < 0) return NULL;
   if (PyArray_AsCArray(&cc_ket_obj, (void *)&cc_ket, dims, 2, descr) < 0) return NULL;

   /* Call external C function to compute values of (m0|n0) integrals in-place, 
      using computed indexing within bra and ket */

   two_electron_contract(contracted_ints, primitive_ints, cc_bra, cc_ket, na, nb, nc, nd, lbra, lket); 

   /* Return none */
   return Py_BuildValue(""); 

}

/*---------------------------------------------------------------------*/

static PyObject *c_ints_two_electron_hrr(PyObject *self, PyObject *args) {

   /* Declare types for objects received from python */
   int la, lb, lc, ld, goofy;
   PyObject *target_ints_obj, *base_int0_obj, *base_int1_obj, *Rx_obj;

   /* C pointers to the arrays contained within those objects */
   double **target_ints, **base_int0, **base_int1, *Rx;

   /* Output variables that capture properties of input arrays */
   npy_intp dims[3];
   int typenum = NPY_DOUBLE;
   PyArray_Descr *descr;
   descr = PyArray_DescrFromType(typenum);

   /* Parse the input tuple */
   if (!PyArg_ParseTuple(args, "OOOOiiiii", &target_ints_obj, &base_int0_obj, 
                         &base_int1_obj, &Rx_obj, &la, &lb, &lc, &ld, &goofy)) 
      {
         PyErr_SetString(PyExc_TypeError, "Error parsing objects passed to C");
         return NULL;
      }
   
   /* Extract pointers to memory locations directly, without copying data to C */
   if (PyArray_AsCArray(&target_ints_obj, (void **)&target_ints, dims, 2, descr) < 0) return NULL;
   if (PyArray_AsCArray(&base_int0_obj, (void **)&base_int0, dims, 2, descr) < 0) return NULL;
   if (PyArray_AsCArray(&base_int1_obj, (void **)&base_int1, dims, 2, descr) < 0) return NULL;
   if (PyArray_AsCArray(&Rx_obj, (void *)&Rx, dims, 1, descr) < 0) return NULL;

   /* Call external C function to compute values of (mn|ls) integrals in-place, 
      using computed indexing within bra and ket */

   two_electron_hrr(target_ints, base_int0, base_int1, Rx, la, lb, lc, ld, goofy);

   /* Return none */
   return Py_BuildValue(""); 

}

/*---------------------------------------------------------------------*/

static PyObject *c_ints_one_electron_fundamentals(PyObject *self, PyObject *args) {

   /* Declare types for objects received from python */
   int na, nb, m;
   double Z;
   PyObject *nuclear_fundamentals_obj, *Rc_obj; 
   PyObject *sigma_P_obj, *U_P_obj, *P_obj; 

   /* C pointers to arrays contained within those objects */
   double ***nuclear_fundamentals;
   double **sigma_P, **U_P, ***P;
   double *Rc;

   /* Output variables that capture properties of input arrays */
   npy_intp dims[3];
   int typenum = NPY_DOUBLE;
   PyArray_Descr *descr;
   descr = PyArray_DescrFromType(typenum);

   /* Parse the input tuple */
   if (!PyArg_ParseTuple(args, "OOOOOdiii", &nuclear_fundamentals_obj, 
                         &sigma_P_obj, &U_P_obj, &P_obj, &Rc_obj,
                         &Z, &na, &nb, &m)) 
      {
         PyErr_SetString(PyExc_TypeError, "Error parsing objects passed to C");
         return NULL;
      }
 
   /* Extract pointers to memory locations directly, without copying data to C */
   if (PyArray_AsCArray(&nuclear_fundamentals_obj, (void ***)&nuclear_fundamentals, dims, 3, descr) < 0) return NULL;
   if (PyArray_AsCArray(&P_obj, (void ***)&P, dims, 3, descr) < 0) return NULL;
   if (PyArray_AsCArray(&sigma_P_obj, (void **)&sigma_P, dims, 2, descr) < 0) return NULL;
   if (PyArray_AsCArray(&U_P_obj, (void **)&U_P, dims, 2, descr) < 0) return NULL;
   if (PyArray_AsCArray(&Rc_obj, (void *)&Rc, dims, 1, descr) < 0) return NULL;

   /* Call external C function to compute values of [m|0] nuclear attraction integrals in-place */

   one_electron_fundamentals(nuclear_fundamentals, sigma_P, U_P, P, Rc, Z, na, nb, m);

   /* Return none */
   return Py_BuildValue(""); 

}

/*---------------------------------------------------------------------*/

static PyObject *c_ints_one_electron_vrr(PyObject *self, PyObject *args) {

   /* Declare types for objects received from python */
   int na, nb, la, n_base_classes, atom_index, aux;
   PyObject *target_ints_obj, *base_ints_obj;
   PyObject *sigma_P_obj, *P_obj; 
   PyObject *Rc_obj, *Ra_obj;

   /* Declare types for sub-objects within base_ints object */
   PyObject *base_int0_obj, *base_int1_obj, *base_int2_obj, *base_int3_obj;

   /* C pointers to arrays contained within objects */
   double **target_ints, **base_int0, **base_int1, **base_int2, **base_int3;
   double **sigma_P, ***P, *Ra, *Rc;

   /* Output variables that capture properties of input arrays */
   npy_intp dims[3];
   int typenum = NPY_DOUBLE;
   PyArray_Descr *descr;
   descr = PyArray_DescrFromType(typenum);

   /* Parse the input tuple */
   if (!PyArg_ParseTuple(args, "OOOOOOiiiii", &target_ints_obj, &base_ints_obj, 
                         &sigma_P_obj, &P_obj, &Ra_obj, &Rc_obj, 
                         &atom_index, &n_base_classes, &na, &nb, &la)) 
      {
         PyErr_SetString(PyExc_TypeError, "Error parsing objects passed to C");
         return NULL;
      }
 
   /* Extract pointers to memory locations directly, without copying data to C */
   if (PyArray_AsCArray(&target_ints_obj, (void **)&target_ints, dims, 2, descr) < 0) return NULL;
   if (PyArray_AsCArray(&P_obj, (void ***)&P, dims, 3, descr) < 0) return NULL;
   if (PyArray_AsCArray(&sigma_P_obj, (void **)&sigma_P, dims, 2, descr) < 0) return NULL;
   if (PyArray_AsCArray(&Ra_obj, (void *)&Ra, dims, 1, descr) < 0) return NULL;
   if (PyArray_AsCArray(&Rc_obj, (void *)&Rc, dims, 1, descr) < 0) return NULL;

   /* Assign pointers to ints within base_ints list, dummy to base_int0 if not supplied */
   base_int0_obj = PyList_GetItem(base_ints_obj, (Py_ssize_t) 0);
   if (PyArray_AsCArray(&base_int0_obj, (void **)&base_int0, dims, 2, descr) < 0) return NULL;
   if (n_base_classes > 1) {
      base_int1_obj = PyList_GetItem(base_ints_obj, (Py_ssize_t) 1);
      if (PyArray_AsCArray(&base_int1_obj, (void **)&base_int1, dims, 2, descr) < 0) return NULL;
   } else {
      if (PyArray_AsCArray(&base_int0_obj, (void **)&base_int1, dims, 2, descr) < 0) return NULL;
   }
   if (n_base_classes == 4) {
      base_int2_obj = PyList_GetItem(base_ints_obj, (Py_ssize_t) 2);
      base_int3_obj = PyList_GetItem(base_ints_obj, (Py_ssize_t) 3);
      if (PyArray_AsCArray(&base_int2_obj, (void **)&base_int2, dims, 2, descr) < 0) return NULL;
      if (PyArray_AsCArray(&base_int3_obj, (void **)&base_int3, dims, 2, descr) < 0) return NULL;
   } else {
      if (PyArray_AsCArray(&base_int0_obj, (void **)&base_int2, dims, 2, descr) < 0) return NULL;
      if (PyArray_AsCArray(&base_int0_obj, (void **)&base_int3, dims, 2, descr) < 0) return NULL;
   }

   /* Use atom_index to determine whether to compute nuclear attraction (aux = 1) or overlap integrals (aux = 0) */
   if (atom_index == -1) { aux = 0; } else { aux = 1; };

   /* Call external C function to compute values of [m|0] nuclear attraction or overlap integrals in-place */

   one_electron_vrr(target_ints, base_int0, base_int1, base_int2, base_int3, sigma_P, P, Ra, Rc, na, nb, la, aux);

   /* Return none */
   return Py_BuildValue(""); 

}

/*---------------------------------------------------------------------*/

static PyObject *c_ints_one_electron_hrr(PyObject *self, PyObject *args) {

   /* Declare types for objects received from python */
   int na, nb, la, lb, n_base_classes;
   PyObject *target_ints_obj, *base_ints_obj, *Rab_obj; 

   /* Declare types for sub-objects within base_ints object */
   PyObject *base_int0_obj, *base_int1_obj;

   /* C pointers to arrays contained within objects */
   double **target_ints, **base_int0, **base_int1, *Rab;

   /* Output variables that capture properties of input arrays */
   npy_intp dims[3];
   int typenum = NPY_DOUBLE;
   PyArray_Descr *descr;
   descr = PyArray_DescrFromType(typenum);

   /* Parse the input tuple */
   if (!PyArg_ParseTuple(args, "OOOiiiii", &target_ints_obj, &base_ints_obj, &Rab_obj,
                         &n_base_classes, &na, &nb, &la, &lb)) 
      {
         PyErr_SetString(PyExc_TypeError, "Error parsing objects passed to C");
         return NULL;
      }
 
   /* Extract pointers to memory locations directly, without copying data to C */
   if (PyArray_AsCArray(&target_ints_obj, (void **)&target_ints, dims, 2, descr) < 0) return NULL;
   if (PyArray_AsCArray(&Rab_obj, (void *)&Rab, dims, 1, descr) < 0) return NULL;

   /* Assign pointers to ints within base_ints list, dummy to base_int0 if not supplied */
   base_int0_obj = PyList_GetItem(base_ints_obj, (Py_ssize_t) 0);
   if (PyArray_AsCArray(&base_int0_obj, (void **)&base_int0, dims, 2, descr) < 0) return NULL;
   if (n_base_classes == 2) {
      base_int1_obj = PyList_GetItem(base_ints_obj, (Py_ssize_t) 1);
      if (PyArray_AsCArray(&base_int1_obj, (void **)&base_int1, dims, 2, descr) < 0) return NULL;
   } else {
      if (PyArray_AsCArray(&base_int0_obj, (void **)&base_int1, dims, 2, descr) < 0) return NULL;
   }

   /* Call external C function to compute values of [m|n] nuclear attraction or overlap integrals in-place */

   one_electron_hrr(target_ints, base_int0, base_int1, Rab, na, nb, la, lb);

   /* Return none */
   return Py_BuildValue(""); 

}

/*---------------------------------------------------------------------*/

static PyObject *c_ints_one_electron_kinetic(PyObject *self, PyObject *args) {

   /* Declare types for objects received from python */
   int na, nb, la, lb, n_base_classes;
   PyObject *target_ints_obj, *base_ints_obj, *exB_obj; 

   /* Declare types for sub-objects within base_ints object */
   PyObject *base_int0_obj, *base_int1_obj, *base_int2_obj;

   /* C pointers to arrays contained within objects */
   double **target_ints, **base_int0, **base_int1, **base_int2, *exB;

   /* Output variables that capture properties of input arrays */
   npy_intp dims[3];
   int typenum = NPY_DOUBLE;
   PyArray_Descr *descr;
   descr = PyArray_DescrFromType(typenum);

   /* Parse the input tuple */
   if (!PyArg_ParseTuple(args, "OOOiiiii", &target_ints_obj, &base_ints_obj, &exB_obj,
                         &n_base_classes, &na, &nb, &la, &lb)) 
      {
         PyErr_SetString(PyExc_TypeError, "Error parsing objects passed to C");
         return NULL;
      }
 
   /* Extract pointers to memory locations directly, without copying data to C */
   if (PyArray_AsCArray(&target_ints_obj, (void **)&target_ints, dims, 2, descr) < 0) return NULL;
   if (PyArray_AsCArray(&exB_obj, (void *)&exB, dims, 1, descr) < 0) return NULL;

   /* Assign pointers to ints within base_ints list, dummy to base_int0 if not supplied */
   base_int0_obj = PyList_GetItem(base_ints_obj, (Py_ssize_t) 0);
   base_int1_obj = PyList_GetItem(base_ints_obj, (Py_ssize_t) 1);
   if (PyArray_AsCArray(&base_int0_obj, (void **)&base_int0, dims, 2, descr) < 0) return NULL;
   if (PyArray_AsCArray(&base_int1_obj, (void **)&base_int1, dims, 2, descr) < 0) return NULL;
   if (n_base_classes == 3) {
      base_int2_obj = PyList_GetItem(base_ints_obj, (Py_ssize_t) 2);
      if (PyArray_AsCArray(&base_int2_obj, (void **)&base_int2, dims, 2, descr) < 0) return NULL;
   } else {
      if (PyArray_AsCArray(&base_int0_obj, (void **)&base_int2, dims, 2, descr) < 0) return NULL;
   }

   /* Call external C function to compute values of [m|n] nuclear attraction or overlap integrals in-place */

   one_electron_kinetic(target_ints, base_int0, base_int1, base_int2, exB, na, nb, la, lb);

   /* Return none */
   return Py_BuildValue(""); 

}

/*---------------------------------------------------------------------*/

static PyObject *c_ints_one_electron_contract(PyObject *self, PyObject *args) {

   /* Declare types for objects received from python */
   int na, nb, la, lb;
   PyObject *target_ints_obj, *base_ints_obj, *cc_obj;

   /* C pointers to arrays contained within objects */
   double **target_ints, **base_ints, **cc;

   /* Output variables that capture properties of input arrays */
   npy_intp dims[3];
   int typenum = NPY_DOUBLE;
   PyArray_Descr *descr;
   descr = PyArray_DescrFromType(typenum);

   /* Parse the input tuple */
   if (!PyArg_ParseTuple(args, "OOOiiii", &target_ints_obj, &base_ints_obj, 
                         &cc_obj, &na, &nb, &la, &lb)) 
      {
         PyErr_SetString(PyExc_TypeError, "Error parsing objects passed to C");
         return NULL;
      }
 
   /* Extract pointers to memory locations directly, without copying data to C */
   if (PyArray_AsCArray(&target_ints_obj, (void **)&target_ints, dims, 2, descr) < 0) return NULL;
   if (PyArray_AsCArray(&base_ints_obj, (void **)&base_ints, dims, 2, descr) < 0) return NULL;
   if (PyArray_AsCArray(&cc_obj, (void **)&cc, dims, 2, descr) < 0) return NULL;

   /* Call external C function to contract [m|n] integrals -> (m|n) */

   one_electron_contract(target_ints, base_ints, cc, na, nb, la, lb);

   /* Return none */
   return Py_BuildValue(""); 

}


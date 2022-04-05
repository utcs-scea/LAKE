/* 
 * File:   proj_types.h
 * Author: i9c
 *
 * Created on 19 de Junho de 2015, 00:52
 */

#ifndef PROJ_TYPES_H
#define	PROJ_TYPES_H

#include <string.h>

#ifdef	__cplusplus
extern "C" {
#endif
    
    
    
#ifndef __linux__
    
#ifndef _UINT64_T
#define _UINT64_T
typedef unsigned long long uint64_t;
#endif /* _UINT64_T */

typedef uint64_t        uint_fast64_t;

typedef unsigned long long ulong;

#endif /*__linux__*/

#ifdef	__cplusplus
}
#endif

#endif	/* PROJ_TYPES_H */


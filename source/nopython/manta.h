/******************************************************************************
 *
 * MantaFlow fluid solver framework
 * Copyright 2011-2014 Tobias Pfaff, Nils Thuerey
 *
 * This program is free software, distributed under the terms of the
 * GNU General Public License (GPL)
 * http://www.gnu.org/licenses
 *
 * No python main include
 *
 ******************************************************************************/

#ifndef _MANTA_H
#define _MANTA_H

// Remove preprocessor keywords, so there won't infere with autocompletion etc.
#define KERNEL(...) extern int i,j,k,idx,X,Y,Z;
#define PYTHON(...)
#define returns(X) extern X;
#define alias typedef

#include "general.h"
#include "vectorbase.h"
#include "pclass.h"
#include "fluidsolver.h"

#endif

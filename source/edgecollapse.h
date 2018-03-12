/******************************************************************************
 *
 * MantaFlow fluid solver framework
 * Copyright 2011 Tobias Pfaff, Nils Thuerey 
 *
 * This program is free software, distributed under the terms of the
 * GNU General Public License (GPL) 
 * http://www.gnu.org/licenses
 *
 * Mesh edge collapse and subdivision
 *
 ******************************************************************************/

/******************************************************************************/
// Copyright note:
//
// These functions (C) Chris Wojtan
// Long-term goal is to unify with his split&merge codebase
//
/******************************************************************************/

#ifndef _EDGECOLLAPSE_H
#define _EDGECOLLAPSE_H

#include "mesh.h"

namespace Manta {

void CollapseEdge(Mesh& mesh, const int trinum, const int which, const Vec3 &edgevect, const Vec3 &endpoint,
                  std::vector<int> &deletedNodes, std::map<int,bool> &taintedTris, int &numCollapses, bool doTubeCutting);

Vec3 ModifiedButterflySubdivision(Mesh& mesh, const Corner& ca, const Corner& cb, const Vec3& fallback);
    
}

#endif
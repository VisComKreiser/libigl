#include "quat_conjugate.h"

template <typename Q_type>
IGL_INLINE void igl::quat_conjugate(
  const Q_type *q1, 
  Q_type *out)
{
  out[0] = -q1[0];
  out[1] = -q1[1];
  out[2] = -q1[2];
  out[3] = q1[3];
}

#ifndef IGL_HEADER_ONLY
// Explicit template specialization
#endif
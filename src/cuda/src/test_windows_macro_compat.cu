// ===== Windows macro compatibility compile fixture =====
// Windows rpcndr.h (pulled in by windows.h without WIN32_LEAN_AND_MEAN, e.g.
// via secure_erase.hpp) defines `small` as `char`, which rewrites any
// identifier named `small` in subsequently-included headers into invalid
// syntax. This broke every Windows GPU-host TU including secp256k1.cuh after
// a Windows header. Pre-defining the macro here makes the collision
// deterministic on ALL platforms: this TU fails to compile if an identifier
// colliding with the rpcndr.h macro set is reintroduced.
#define small char

#include "secp256k1.cuh"

int main()
{
    return 0;
}

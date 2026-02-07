#include <secp256k1/field.hpp>
#include <iostream>
#include <iomanip>

using namespace secp256k1::fast;

int main() {
    std::cout << "Testing multiplication..." << std::endl;
    
    FieldElement a = FieldElement::from_uint64(2);
    FieldElement b = FieldElement::from_uint64(3);
    
    std::cout << "2 = " << a.to_hex().substr(0, 16) << "..." << std::endl;
    std::cout << "3 = " << b.to_hex().substr(0, 16) << "..." << std::endl;
    
    std::cout << "Computing 2 * 3..." << std::endl;
    FieldElement result = a * b;
    
    std::cout << "Result = " << result.to_hex() << std::endl;
    
    FieldElement expected = FieldElement::from_uint64(6);
    if (result == expected) {
        std::cout << "✓ Test passed: 2 * 3 = 6" << std::endl;
        return 0;
    } else {
        std::cout << "✗ Test failed: expected 6, got " << result.to_hex() << std::endl;
        return 1;
    }
}

#ifndef SECP256K1_SORTED_DB_ADAPTER_HPP
#define SECP256K1_SORTED_DB_ADAPTER_HPP

#include "sorted_db/sorted_db.hpp"
#include "secp256k1/field.hpp"
#include <vector>
#include <cstring>

namespace secp256k1::fast {

/**
 * SortedDBAdapter - Adapter to use sorted_db with secp256k1 FieldElements
 * 
 * Provides the same lookup API but uses binary search on sorted file.
 * Drop-in replacement for maximum performance.
 */
class SortedDBAdapter {
public:
    explicit SortedDBAdapter(const std::string& db_path) 
        : db_(db_path) {}
    
    ~SortedDBAdapter() = default;
    
    // Non-copyable
    SortedDBAdapter(const SortedDBAdapter&) = delete;
    SortedDBAdapter& operator=(const SortedDBAdapter&) = delete;
    
    // Movable
    SortedDBAdapter(SortedDBAdapter&&) noexcept = default;
    SortedDBAdapter& operator=(SortedDBAdapter&&) noexcept = default;
    
    /**
     * Fast point lookup using raw bytes (compatible with legacy lookup API)
     * @param x Field element (X coordinate) to search for
     * @return true if found, false otherwise
     */
    bool lookup_raw(const FieldElement& x) {
        // Convert FieldElement to 32-byte key
        auto bytes = x.to_bytes();
        
        // Reinterpret as uint64_t[4] for sorted_db
        uint64_t key[4];
        std::memcpy(key, bytes.data(), 32);
        
        return db_.lookup(key);
    }
    
    /**
     * Batch lookup using raw bytes (compatible with legacy lookup API)
     * @param x_coords Array of X coordinates to search for
     * @param count Number of elements in array
     * @param results Output array of booleans (true if found)
     */
    void lookup_raw_batch(const FieldElement* x_coords, size_t count, uint8_t* results) {
        // Convert all FieldElements to keys
        std::vector<uint64_t> keys(count * 4);
        
        for (size_t i = 0; i < count; i++) {
            auto bytes = x_coords[i].to_bytes();
            std::memcpy(&keys[i * 4], bytes.data(), 32);
        }
        
        // Batch lookup
        db_.lookup_batch(keys.data(), count, results);
    }
    
    /**
     * Get approximate number of keys in database
     * @return Entry count
     */
    uint64_t estimate_num_keys() const {
        return db_.get_entry_count();
    }
    
    /**
     * Check if database is open and valid
     * @return true if database is ready for lookups
     */
    bool is_open() const noexcept {
        return db_.is_open();
    }
    
    /**
     * Get statistics (compatible with legacy lookup API)
     * @return Statistics string with performance metrics
     */
    std::string get_statistics() const {
        return db_.get_statistics();
    }
    
private:
    sorted_db::SortedDBLookup db_;
};

} // namespace secp256k1::fast

#endif // SECP256K1_SORTED_DB_ADAPTER_HPP

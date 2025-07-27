#pragma once

#include "launcher.h"
#include "sorting_common.h"
#include "loadimpl.h"

template <
    typename KeyT,
    int BLOCK_THREADS_,
    int WARP_SIZE_,
    int KEYS_PER_THREAD_,
    bool IS_DESCENDING_ = false,
    typename ValueT = NullType,
    typename DigitT = uint16_t,   // Covering BLOCK_THREADS * KEYS_PER_THREAD.
    typename CounterT = uint32_t, // Packed scan datatype
    // We are going to bundle multiple counters with 'DigitT' type to perform
    // packed prefix sum.
    int RADIX_BITS_ = 4>
class BlockRadixSort {
public:
    using KeyTraitsT = typename KeyTraits<KeyT>::Type;

    enum {
        BLOCK_THREADS = BLOCK_THREADS_,
        WARP_SIZE = WARP_SIZE_,
        KEYS_PER_THREAD = KEYS_PER_THREAD_,
        IS_DESCENDING = IS_DESCENDING_,
        RADIX_BITS = RADIX_BITS_,

        PROCESSING_LENGTH = BLOCK_THREADS * KEYS_PER_THREAD,
        RADIX_BUCKETS = 1 << RADIX_BITS,
        KEYS_ONLY = std::is_same<ValueT, NullType>::value,
        PACKING_RATIO = sizeof(CounterT) / sizeof(DigitT),
        COUNTER_LANES = RADIX_BUCKETS / PACKING_RATIO,
        LOG_COUNTER_LANES = Log2<COUNTER_LANES>::VALUE,
        DIGIT_BITS = sizeof(DigitT) << 3,
        DIGIT_MASK = (1 << DIGIT_BITS) - 1,
        IS_INT_TYPE = std::is_integral<ValueT>::value,
    };

    static_assert(sizeof(CounterT) >= sizeof(DigitT), "");
    static_assert(sizeof(CounterT) % sizeof(DigitT) == 0, "");
    static_assert(
        ((1l << (sizeof(DigitT) << 3)) - 1) >= (BLOCK_THREADS * KEYS_PER_THREAD),
        " ");

private:
    union RankT {
        CounterT counters[COUNTER_LANES][BLOCK_THREADS];
        CounterT counters_flat[COUNTER_LANES * BLOCK_THREADS];
        DigitT buckets[COUNTER_LANES][BLOCK_THREADS][PACKING_RATIO];
    };

    union LocalStorage {
        RankT rank_storage;
        struct {
            KeyTraitsT exchange_ukeys[PROCESSING_LENGTH];
            int relative_bin_offsets[RADIX_BUCKETS];
        };
        ValueT exchange_values[PROCESSING_LENGTH];
    };

    ITEM &item_;
    LocalStorage *local_storage_;
    int lid_;
    int bin_offset_;

    int ranks_[KEYS_PER_THREAD];
    KeyTraitsT ukeys_[KEYS_PER_THREAD];
    ValueT values_[KEYS_PER_THREAD];
    int relative_bin_offsets_[KEYS_PER_THREAD];
    int begin_bit_;
    int pass_bits_;
    bool enable_bin_offsets_ = false;

public:
    static int LocalMemorySize() {
        return sizeof(LocalStorage);
    }

    DEVICE inline void load_bin_offsets(int *counts, int block_id, int num_blocks) {
        int bin_idx = lid_;
        if (lid_ < RADIX_BUCKETS) {
            if (IS_DESCENDING)
                bin_idx = RADIX_BUCKETS - bin_idx - 1;
            bin_offset_ = counts[block_id + bin_idx * num_blocks];
        }
        enable_bin_offsets_ = true;
        item_.barrier();
    }

    DEVICE inline BlockRadixSort(ITEM &item) :
        item_(item),
        local_storage_(reinterpret_cast<LocalStorage *>(item.shared_ptr())),
        lid_(item.thread_idx_x()) {
    }

    DEVICE inline void load_keys(
        const KeyT *keys_block_in,
        int num_elements,
        int block_offset = 0) {
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            int offset = block_offset + lid_ * KEYS_PER_THREAD + ITEM;
            if (offset < num_elements) {
                ukeys_[ITEM] =
                    KeyTraits<KeyT>::convert(loadimpl::load(&keys_block_in[offset]));
            } else {
                KeyTraitsT padding_key;
                if (IS_DESCENDING) {
                    padding_key = 0;
                } else {
                    constexpr uint64_t KEY_TRAITS_TYPE_MASK = (uint64_t)1
                                                              << ((sizeof(KeyTraitsT) << 3) - 1);
                    padding_key = static_cast<KeyTraitsT>(KEY_TRAITS_TYPE_MASK);
                    padding_key = padding_key ^ (padding_key - 1);
                }
                ukeys_[ITEM] = padding_key;
            }
        }
    }

    DEVICE inline void load_values(
        const ValueT *values_block_in,
        int num_elements,
        int block_offset = 0) {
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            int offset = block_offset + lid_ * KEYS_PER_THREAD + ITEM;
            if (offset < num_elements) {
                if constexpr (IS_INT_TYPE) {
                    values_[ITEM] =
                        values_block_in == nullptr ? offset : values_block_in[offset];
                } else {
                    values_[ITEM] = values_block_in[offset];
                }
            }
        }
    }

    DEVICE inline void store_keys(KeyT *keys_block_out, int num_elements) {
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            local_storage_->exchange_ukeys[lid_ * KEYS_PER_THREAD + ITEM] =
                ukeys_[ITEM];
        }
        item_.barrier();
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            int offset = lid_ + ITEM * BLOCK_THREADS;
            if (offset < num_elements) {
                keys_block_out[offset] = KeyTraits<KeyT>::deconvert(local_storage_->exchange_ukeys[offset]);
            }
        }
        item_.barrier();
    }

    DEVICE inline void store_keys(KeyT *out, int offset_select, int num_selected) {
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            if (ranks_[ITEM] < offset_select) {
                auto key = KeyTraits<KeyT>::deconvert(ukeys_[ITEM]);
                out[num_selected + ranks_[ITEM]] = key;
            }
        }
    }

    DEVICE inline void store_values(ValueT *values_block_out, int num_elements) {
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            local_storage_->exchange_values[lid_ * KEYS_PER_THREAD + ITEM] =
                values_[ITEM];
        }
        item_.barrier();
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            int offset = lid_ + ITEM * BLOCK_THREADS;
            if (offset < num_elements) {
                values_block_out[offset] = local_storage_->exchange_values[offset];
            }
        }
        item_.barrier();
    }

    DEVICE inline void store_values(ValueT *out, int offset_select, int num_selected) {
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            if (ranks_[ITEM] < offset_select) {
                out[num_selected + ranks_[ITEM]] = values_[ITEM];
            }
        }
    }

    DEVICE inline void exchange_and_store_keys(KeyT *keys_out, int num_elements) {
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            local_storage_->exchange_ukeys[ranks_[ITEM]] = ukeys_[ITEM];
        }
        item_.barrier();
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            int offset = lid_ + ITEM * BLOCK_THREADS;
            auto ukey = local_storage_->exchange_ukeys[offset];
            relative_bin_offsets_[ITEM] =
                local_storage_->relative_bin_offsets[extract_digit(ukey)];
            offset += relative_bin_offsets_[ITEM];
            if (offset < num_elements) {
                keys_out[offset] = KeyTraits<KeyT>::deconvert(ukey);
            }
        }
        item_.barrier();
    }

    DEVICE inline void exchange_and_store_values(ValueT *values_out, int num_elements) {
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            local_storage_->exchange_values[ranks_[ITEM]] = values_[ITEM];
        }
        item_.barrier();
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            int offset = lid_ + ITEM * BLOCK_THREADS;
            auto value = local_storage_->exchange_values[offset];
            offset += relative_bin_offsets_[ITEM];
            if (offset < num_elements) {
                values_out[offset] = value;
            }
        }
        item_.barrier();
    }

    DEVICE inline void exchange_keys() {
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            local_storage_->exchange_ukeys[ranks_[ITEM]] = ukeys_[ITEM];
        }
        item_.barrier();
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            int offset = lid_ * KEYS_PER_THREAD + ITEM;
            ukeys_[ITEM] = local_storage_->exchange_ukeys[offset];
        }
        item_.barrier();
    }

    DEVICE inline void exchange_keys(
        int lower_offset,
        int upper_offset,
        uint32_t *mask) {
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            if (ranks_[ITEM] >= lower_offset && ranks_[ITEM] < upper_offset) {
                local_storage_->exchange_ukeys[ranks_[ITEM] - lower_offset] =
                    ukeys_[ITEM];
            }
        }
        item_.barrier();
        *mask = 0u;
        int new_length = upper_offset - lower_offset;
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            int offset = lid_ * KEYS_PER_THREAD + ITEM;
            if (offset < new_length) {
                *mask |= (1u << ITEM);
                ukeys_[ITEM] = local_storage_->exchange_ukeys[offset];
            }
        }
        item_.barrier();
    }

    DEVICE inline void exchange_values() {
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            local_storage_->exchange_values[ranks_[ITEM]] = values_[ITEM];
        }
        item_.barrier();
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            int offset = lid_ * KEYS_PER_THREAD + ITEM;
            values_[ITEM] = local_storage_->exchange_values[offset];
        }
        item_.barrier();
    }

    DEVICE inline void exchange_values(int lower_offset, int upper_offset) {
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            if (ranks_[ITEM] >= lower_offset && ranks_[ITEM] < upper_offset) {
                local_storage_->exchange_values[ranks_[ITEM] - lower_offset] =
                    values_[ITEM];
            }
        }
        item_.barrier();
        int new_length = upper_offset - lower_offset;
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            int offset = lid_ * KEYS_PER_THREAD + ITEM;
            if (offset < new_length) {
                values_[ITEM] = local_storage_->exchange_values[offset];
            }
        }
        item_.barrier();
    }

    DEVICE inline DigitT extract_digit(KeyTraitsT key) {
        return ((key >> begin_bit_) & ((1 << pass_bits_) - 1));
    }

    DEVICE inline void rank_keys(int begin_bit, int end_bit) {
        begin_bit_ = begin_bit;
        pass_bits_ = end_bit - begin_bit_;
        pass_bits_ = RADIX_BITS < pass_bits_ ? RADIX_BITS : pass_bits_;
        DigitT *digit_counters[KEYS_PER_THREAD];

        // reset buckets
#pragma unroll
        for (int ITEM = 0; ITEM < COUNTER_LANES; ++ITEM) {
            local_storage_->rank_storage.counters[ITEM][lid_] = 0;
        }
        item_.barrier();

#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            auto digit = extract_digit(ukeys_[ITEM]);
            auto sub_counter = digit >> LOG_COUNTER_LANES;
            auto counter_lane = digit & (COUNTER_LANES - 1);
            if (IS_DESCENDING) {
                sub_counter = PACKING_RATIO - 1 - sub_counter;
                counter_lane = COUNTER_LANES - 1 - counter_lane;
            }
            digit_counters[ITEM] =
                &local_storage_->rank_storage.buckets[counter_lane][lid_][sub_counter];
            ranks_[ITEM] = *digit_counters[ITEM];
            *digit_counters[ITEM] = ranks_[ITEM] + 1;
        }
        item_.barrier();

        CounterT exclusive = block_exclusive_cumsum<
            CounterT,
            COUNTER_LANES,
            BLOCK_THREADS,
            WARP_SIZE>(local_storage_->rank_storage.counters_flat, item_);

        CounterT c = 0;
#pragma unroll
        for (int STEP = 1; STEP < PACKING_RATIO; ++STEP) {
            exclusive = exclusive << DIGIT_BITS;
            c += exclusive;
        }

#pragma unroll
        for (int INDEX = 0; INDEX < COUNTER_LANES; ++INDEX) {
            local_storage_->rank_storage.counters[INDEX][lid_] += c;
        }
        item_.barrier();

        // inc rank
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            ranks_[ITEM] += *digit_counters[ITEM];
        }
        item_.barrier();

        if (enable_bin_offsets_) {
            int digit = lid_;
            if (lid_ < RADIX_BUCKETS) {
                if (IS_DESCENDING)
                    digit = RADIX_BUCKETS - digit - 1;
                auto sub_counter = digit >> LOG_COUNTER_LANES;
                auto counter_lane = digit & (COUNTER_LANES - 1);
                int digit_offset =
                    local_storage_->rank_storage.buckets[counter_lane][0][sub_counter];
                local_storage_->relative_bin_offsets[lid_] = bin_offset_ - digit_offset;
            }
            item_.barrier();
        }
    }

    DEVICE inline void find_select_offset(
        int carry,
        int num_to_select,
        int *out_offset_select,
        int *out_offset_active) {
        *out_offset_select = 0;
        *out_offset_active = 0;
#pragma unroll
        for (int DIGIT = 1; DIGIT < RADIX_BUCKETS; ++DIGIT) {
            auto sub_counter = DIGIT >> LOG_COUNTER_LANES;
            auto counter_lane = DIGIT & (COUNTER_LANES - 1);
            auto count = (int)(local_storage_->rank_storage
                                   .buckets[counter_lane][0][sub_counter]);
            if (count > num_to_select) {
                *out_offset_active = count;
                break;
            }
            *out_offset_select = count;
        }
        if (*out_offset_active == 0)
            *out_offset_active = carry;
    }

    DEVICE inline void rank_keys(
        int begin_bit,
        int pass_bits,
        uint32_t active_mask,
        int num_to_select,
        int *out_offset_select,
        int *out_offset_active) {
        begin_bit_ = begin_bit;
        pass_bits_ = pass_bits;
        DigitT *digit_counters[KEYS_PER_THREAD];

        // reset buckets
#pragma unroll
        for (int ITEM = 0; ITEM < COUNTER_LANES; ++ITEM) {
            local_storage_->rank_storage.counters[ITEM][lid_] = 0;
        }
        item_.barrier();

#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            ranks_[ITEM] = PROCESSING_LENGTH;
            if (active_mask >> ITEM & 1) {
                auto digit = extract_digit(ukeys_[ITEM]);
                auto sub_counter = digit >> LOG_COUNTER_LANES;
                auto counter_lane = digit & (COUNTER_LANES - 1);
                if (IS_DESCENDING) {
                    sub_counter = PACKING_RATIO - 1 - sub_counter;
                    counter_lane = COUNTER_LANES - 1 - counter_lane;
                }
                digit_counters[ITEM] = &local_storage_->rank_storage
                                            .buckets[counter_lane][lid_][sub_counter];
                ranks_[ITEM] = *digit_counters[ITEM];
                *digit_counters[ITEM] = ranks_[ITEM] + 1;
            }
        }
        item_.barrier();

        CounterT exclusive = block_exclusive_cumsum<
            CounterT,
            COUNTER_LANES,
            BLOCK_THREADS,
            WARP_SIZE>(local_storage_->rank_storage.counters_flat, item_);

        int carry = 0;
#pragma unroll
        for (int STEP = 0; STEP < PACKING_RATIO; ++STEP) {
            DigitT cc = (exclusive >> (STEP * DIGIT_BITS)) & DIGIT_MASK;
            carry += cc;
        }

        CounterT c = 0;
#pragma unroll
        for (int STEP = 1; STEP < PACKING_RATIO; ++STEP) {
            exclusive = exclusive << DIGIT_BITS;
            c += exclusive;
        }

#pragma unroll
        for (int INDEX = 0; INDEX < COUNTER_LANES; ++INDEX) {
            local_storage_->rank_storage.counters[INDEX][lid_] += c;
        }
        item_.barrier();

        // inc rank
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            ranks_[ITEM] += *digit_counters[ITEM];
        }
        item_.barrier();

        find_select_offset(
            carry, num_to_select, out_offset_select, out_offset_active);

        item_.barrier();
    }

    DEVICE inline void topk(
        int begin_bit,
        int end_bit,
        int k,
        KeyT *out_keys,
        ValueT *out_values) {
        uint32_t active_mask = 0xffffffff;
        int num_selected = 0;
        while (true) {
            int pass_bits = begin_bit - end_bit;
            pass_bits = pass_bits < RADIX_BITS ? pass_bits : RADIX_BITS;
            begin_bit -= pass_bits;
            int offset_select, offset_active;
            rank_keys(
                begin_bit,
                pass_bits,
                active_mask,
                k - num_selected,
                &offset_select,
                &offset_active);
            if (begin_bit == end_bit)
                offset_select = k - num_selected;
            if (offset_select > 0) {
                store_keys(out_keys, offset_select, num_selected);
                if (!KEYS_ONLY)
                    store_values(out_values, offset_select, num_selected);
            }
            num_selected += offset_select;
            if (num_selected == k)
                break;
            exchange_keys(offset_select, offset_active, &active_mask);
            if (!KEYS_ONLY)
                exchange_values(offset_select, offset_active);
        }
    }

    DEVICE inline void topk_append_keys(
        const KeyT *keys_in,
        const KeyT *keys_temp,
        int num_elements,
        int num_start,
        int k) {
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            int offset = lid_ * KEYS_PER_THREAD + ITEM;
            if (offset < k) {
                ukeys_[ITEM] = KeyTraits<KeyT>::convert(loadimpl::load(&keys_temp[offset]));
            } else {
                offset += num_start - k;
                if (offset < num_elements) {
                    ukeys_[ITEM] = KeyTraits<KeyT>::convert(loadimpl::load(&keys_in[offset]));
                } else {
                    KeyTraitsT padding_key;
                    if (IS_DESCENDING) {
                        padding_key = 0;
                    } else {
                        constexpr uint64_t KEY_TRAITS_TYPE_MASK = (uint64_t)1
                                                                  << ((sizeof(KeyTraitsT) << 3) - 1);
                        padding_key = static_cast<KeyTraitsT>(KEY_TRAITS_TYPE_MASK);
                        padding_key = padding_key ^ (padding_key - 1);
                    }
                    ukeys_[ITEM] = padding_key;
                }
            }
        }
    }

    DEVICE inline void topk_append_values(
        const ValueT *values_in,
        const ValueT *values_temp,
        int num_elements,
        int num_start,
        int k) {
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            int offset = lid_ * KEYS_PER_THREAD + ITEM;
            if (offset < k) {
                values_[ITEM] = values_temp[offset];
            } else {
                offset += num_start - k;
                if (offset < num_elements) {
                    if constexpr (IS_INT_TYPE) {
                        values_[ITEM] = values_in == nullptr ? offset : values_in[offset];
                    } else {
                        values_[ITEM] = values_in[offset];
                    }
                }
            }
        }
    }
};

template <
    typename KeyT,
    int BLOCK_THREADS_,
    int WARP_SIZE_,
    int KEYS_PER_THREAD_,
    bool IS_DESCENDING_ = false,
    typename ValueT = NullType,
    typename DigitT = unsigned char,
    typename CounterT = uint32_t, // Packed scan datatype
    // We are going to bundle multiple counters with 'DigitT' type to perform
    // packed prefix sum.
    int RADIX_BITS = 4>
class RadixSortUpsweep {
public:
    using KeyTraitsT = typename KeyTraits<KeyT>::Type;
    enum {
        BLOCK_THREADS = BLOCK_THREADS_,
        WARP_SIZE = WARP_SIZE_,
        KEYS_PER_THREAD = KEYS_PER_THREAD_,
        IS_DESCENDING = IS_DESCENDING_,

        PROCESSING_LENGTH = BLOCK_THREADS * KEYS_PER_THREAD,
        RADIX_BUCKETS = 1 << RADIX_BITS,
        KEYS_ONLY = std::is_same<ValueT, NullType>::value,
        PACKING_RATIO = sizeof(CounterT) / sizeof(DigitT),
        LOG_PACKING_RATIO = Log2<PACKING_RATIO>::VALUE,
        COUNTER_LANES = RADIX_BUCKETS / PACKING_RATIO,

        WARPS = (BLOCK_THREADS + WARP_SIZE - 1) / WARP_SIZE,
        LANES_PER_WARP =
            std::max<int>(1, (COUNTER_LANES + WARPS - 1) / WARPS),
    };

    static_assert(sizeof(CounterT) >= sizeof(DigitT), "");
    static_assert(sizeof(CounterT) % sizeof(DigitT) == 0, "");

private:
    union LocalStorage {
        CounterT counters[COUNTER_LANES][BLOCK_THREADS];
        DigitT buckets[COUNTER_LANES][BLOCK_THREADS][PACKING_RATIO];
        int block_counters[WARP_SIZE][RADIX_BUCKETS];
    };

    ITEM &item_;
    const KeyT *keys_in_;
    int lid_;
    int gid_;
    int begin_bit_;
    int end_bit_;
    int num_blocks_;
    int *count_out_;
    int warp_id_;
    int warp_tid_;

    LocalStorage *local_storage_;
    int local_counts_[LANES_PER_WARP][PACKING_RATIO];

public:
    static int LocalMemorySize() {
        return sizeof(LocalStorage);
    }

    DEVICE inline RadixSortUpsweep(
        ITEM &item,
        const KeyT *keys_in,
        int gid,
        int begin_bit,
        int end_bit,
        int num_blocks,
        int *count_out) :
        item_(item),
        keys_in_(keys_in),
        lid_(item.thread_idx_x()),
        gid_(gid),
        begin_bit_(begin_bit),
        end_bit_(end_bit),
        num_blocks_(num_blocks),
        count_out_(count_out),
        local_storage_(reinterpret_cast<LocalStorage *>(item.shared_ptr())) {
        warp_id_ = lid_ / WARP_SIZE;
        warp_tid_ = lid_ % WARP_SIZE;
    }

    DEVICE inline DigitT extract_digit(KeyTraitsT key) {
        auto pass_bits = end_bit_ - begin_bit_;
        pass_bits = RADIX_BITS < pass_bits ? RADIX_BITS : pass_bits;
        return ((key >> begin_bit_) & ((1 << pass_bits) - 1));
    }

    DEVICE inline void process_full_tile(int block_offset) {
        KeyTraitsT keys[KEYS_PER_THREAD];
        auto block_ptr = keys_in_ + block_offset;
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            keys[ITEM] = KeyTraits<KeyT>::convert(
                loadimpl::load(&block_ptr[lid_ + ITEM * BLOCK_THREADS]));
        }
        item_.barrier();
#pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM) {
            auto digit = extract_digit(keys[ITEM]);
            auto sub_counter = digit & (PACKING_RATIO - 1);
            auto row_offset = digit >> LOG_PACKING_RATIO;
            local_storage_->buckets[row_offset][lid_][sub_counter]++;
        }
    }

    DEVICE inline void process_partial_tile(int block_offset, int block_end) {
        for (int offset = block_offset + lid_; offset < block_end;
             offset += BLOCK_THREADS) {
            KeyTraitsT key = KeyTraits<KeyT>::convert(loadimpl::load(&keys_in_[offset]));
            auto digit = extract_digit(key);
            auto sub_counter = digit & (PACKING_RATIO - 1);
            auto row_offset = digit >> LOG_PACKING_RATIO;
            local_storage_->buckets[row_offset][lid_][sub_counter]++;
        }
    }

    DEVICE inline void reset_digit_counters() {
#pragma unroll
        for (int LANE = 0; LANE < COUNTER_LANES; ++LANE)
            local_storage_->counters[LANE][lid_] = 0;
    }

    DEVICE inline void reset_unpacked_counters() {
#pragma unroll
        for (int LANE = 0; LANE < LANES_PER_WARP; ++LANE) {
#pragma unroll
            for (int UNPACKED_COUNTER = 0; UNPACKED_COUNTER < PACKING_RATIO;
                 ++UNPACKED_COUNTER) {
                local_counts_[LANE][UNPACKED_COUNTER] = 0;
            }
        }
    }

    DEVICE inline void unpack_digit_counts() {
#pragma unroll
        for (int LANE = 0; LANE < LANES_PER_WARP; ++LANE) {
            int counter_lane = (LANE * WARPS) + warp_id_;
            if (counter_lane < COUNTER_LANES) {
#pragma unroll
                for (int PACKED_COUNTER = 0; PACKED_COUNTER < BLOCK_THREADS;
                     PACKED_COUNTER += WARP_SIZE) {
#pragma unroll
                    for (int UNPACKED_COUNTER = 0; UNPACKED_COUNTER < PACKING_RATIO;
                         ++UNPACKED_COUNTER) {
                        int counter =
                            local_storage_->buckets[counter_lane][warp_tid_ + PACKED_COUNTER]
                                                   [UNPACKED_COUNTER];
                        local_counts_[LANE][UNPACKED_COUNTER] += counter;
                    }
                }
            }
        }
    }

    DEVICE inline void extract_counts() {
#pragma unroll
        for (int LANE = 0; LANE < LANES_PER_WARP; ++LANE) {
            int counter_lane = (LANE * WARPS) + warp_id_;
            if (counter_lane < COUNTER_LANES) {
                int digit_row = counter_lane << LOG_PACKING_RATIO;
#pragma unroll
                for (int UNPACKED_COUNTER = 0; UNPACKED_COUNTER < PACKING_RATIO;
                     ++UNPACKED_COUNTER) {
                    int bin_idx = digit_row + UNPACKED_COUNTER;
                    local_storage_->block_counters[warp_tid_][bin_idx] =
                        local_counts_[LANE][UNPACKED_COUNTER];
                }
            }
        }

        item_.barrier();

        if ((RADIX_BUCKETS % BLOCK_THREADS != 0) && (lid_ < RADIX_BUCKETS)) {
            int bin_idx = lid_;
            int bin_count = 0;
#pragma unroll
            for (int i = 0; i < WARP_SIZE; ++i)
                bin_count += local_storage_->block_counters[i][bin_idx];
            if (IS_DESCENDING)
                bin_idx = RADIX_BUCKETS - bin_idx - 1;
            count_out_[(num_blocks_ * bin_idx) + gid_] = bin_count;
        }
    }

    DEVICE inline void run(int block_offset, int block_end) {
        reset_digit_counters();
        reset_unpacked_counters();

        // Unroll batches of full tiles
        int UNROLL_COUNT = 255 / 4; // the largest value for counter
        int UNROLLED_ELEMENTS = UNROLL_COUNT * PROCESSING_LENGTH;
        while (block_offset + UNROLLED_ELEMENTS <= block_end) {
            for (int i = 0; i < UNROLL_COUNT; ++i) {
                process_full_tile(block_offset);
                block_offset += PROCESSING_LENGTH;
            }
            item_.barrier();
            unpack_digit_counts();
            item_.barrier();
            reset_digit_counters();
        }

        while (block_offset + PROCESSING_LENGTH <= block_end) {
            process_full_tile(block_offset);
            block_offset += PROCESSING_LENGTH;
        }

        process_partial_tile(block_offset, block_end);
        item_.barrier();
        unpack_digit_counts();
        item_.barrier();
        extract_counts();
    }
};

template <int BLOCK_THREADS, int THREAD_WORK_SIZE, int WARP_SIZE_>
class RadixSortScanBins {
public:
    enum {
        WARP_SIZE = WARP_SIZE_,
        PROCESSING_LENGTH = BLOCK_THREADS * THREAD_WORK_SIZE,
        NUM_WARPS = BLOCK_THREADS / WARP_SIZE,
    };

private:
    ITEM &item_;
    int *count_;
    int *slm_;
    int lid_;

public:
    static int LocalMemorySize() {
        return NUM_WARPS * sizeof(int);
    }

    DEVICE inline RadixSortScanBins(
        ITEM &item,
        int *count) :
        item_(item),
        count_(count),
        slm_(reinterpret_cast<int *>(item.shared_ptr())),
        lid_(item.thread_idx_x()) {
    }

    template <bool is_partial>
    DEVICE inline void consume_tile(
        int block_offset,
        int &running_prefix,
        int tile_bound = 0) {
        // Load
        int partial_output[THREAD_WORK_SIZE];
        auto d_local = count_ + block_offset;
#pragma unroll
        for (int ITEM = 0; ITEM < THREAD_WORK_SIZE; ++ITEM) {
            int offset = lid_ * THREAD_WORK_SIZE + ITEM;
            if constexpr (is_partial) {
                if (offset < tile_bound) {
                    partial_output[ITEM] = d_local[offset];
                } else {
                    partial_output[ITEM] = *d_local;
                }
            } else {
                partial_output[ITEM] = d_local[offset];
            }
        }
        item_.barrier();
        // Thread reduce
        int thread_partial = partial_output[0];
#pragma unroll
        for (int ITEM = 1; ITEM < THREAD_WORK_SIZE; ++ITEM) {
            thread_partial = thread_partial + partial_output[ITEM];
        }
        // Warp scan
        int warp_tid = lid_ % WARP_SIZE;
        int warp_id = lid_ / WARP_SIZE;
        const int WARP_SCAN_STEPS = Log2<WARP_SIZE>::VALUE;
        int warp_inclusive_sum, warp_exclusive_sum;
        warp_cumsum<int, WARP_SCAN_STEPS>(
            item_,
            warp_tid,
            thread_partial,
            warp_inclusive_sum,
            warp_exclusive_sum);
        if (warp_tid == (WARP_SIZE - 1))
            slm_[warp_id] = warp_inclusive_sum;
        item_.barrier();
        // Block scan
        int block_all_sum = 0, warp_prefix_sum;
#pragma unroll
        for (int i = 0; i < NUM_WARPS; ++i) {
            if (warp_id == i)
                warp_prefix_sum = block_all_sum;
            block_all_sum += slm_[i];
        }
        warp_exclusive_sum += warp_prefix_sum;
        warp_exclusive_sum += running_prefix;
        running_prefix += block_all_sum;
        // Write back
        int inclusive = partial_output[0];
        inclusive = warp_exclusive_sum + inclusive;
        partial_output[0] = warp_exclusive_sum;
        int exclusive = inclusive;
#pragma unroll
        for (int ITEM = 1; ITEM < THREAD_WORK_SIZE; ++ITEM) {
            inclusive = exclusive + partial_output[ITEM];
            partial_output[ITEM] = exclusive;
            exclusive = inclusive;
        }
#pragma unroll
        for (int ITEM = 0; ITEM < THREAD_WORK_SIZE; ITEM++) {
            int offset = lid_ * THREAD_WORK_SIZE + ITEM;
            if constexpr (is_partial) {
                if (offset < tile_bound) {
                    d_local[offset] = partial_output[ITEM];
                }
            } else {
                d_local[offset] = partial_output[ITEM];
            }
        }
    }

    DEVICE inline void run(int num_counts) {
        int block_offset = 0;
        int running_prefix = 0;
        while (block_offset + PROCESSING_LENGTH <= num_counts) {
            consume_tile<false>(block_offset, running_prefix);
            block_offset += PROCESSING_LENGTH;
        }
        if (block_offset < num_counts) {
            consume_tile<true>(
                block_offset, running_prefix, num_counts - block_offset);
        }
    }
};

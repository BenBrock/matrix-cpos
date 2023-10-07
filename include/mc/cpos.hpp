#pragma once

#include <mc/ranges.hpp>

namespace mc {

namespace __detail {

template <typename T>
concept has_rows_method = requires(T& t) {
  { t.rows() } -> __ranges::forward_range;
  { t.rows() } -> __ranges::view;
};

template <typename T>
concept has_columns_method = requires(T& t) {
  { t.columns() } -> __ranges::forward_range;
  { t.columns() } -> __ranges::view;
};

template <typename T>
concept has_diagonals_method = requires(T& t) {
  { t.diagonals() } -> __ranges::forward_range;
  { t.diagonals() } -> __ranges::view;
};

struct rows_fn_ {
  template <__detail::has_rows_method T>
  constexpr auto operator()(T&& t) const {
    return t.rows();
  }
};

struct columns_fn_ {
  template <__detail::has_columns_method T>
  constexpr auto operator()(T&& t) const {
    return t.columns();
  }
};

struct diagonals_fn_ {
  template <__detail::has_diagonals_method T>
  constexpr auto operator()(T&& t) const {
    return t.diagonals();
  }
};

} // namespace __detail

inline constexpr auto rows = __detail::rows_fn_{};
inline constexpr auto columns = __detail::columns_fn_{};
inline constexpr auto diagonals = __detail::diagonals_fn_{};

template <typename T>
concept row_iterable = requires(T& r) { rows(r); };

template <typename T>
concept column_iterable = requires(T& r) { columns(r); };

template <typename T>
concept diagonal_iterable = requires(T& r) { diagonals(r); };

} // namespace mc

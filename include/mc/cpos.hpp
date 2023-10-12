#pragma once

#include <mc/detail/tag_invoke.hpp>
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

} // namespace __detail

struct rows_fn_ {
  template <__detail::has_rows_method T>
    requires(!mc::is_tag_invocable_v<rows_fn_, T>)
  constexpr auto operator()(T&& t) const {
    return std::forward<T>(t).rows();
  }

  template <typename T>
    requires(mc::is_tag_invocable_v<rows_fn_, T>)
  constexpr auto operator()(T&& t) const {
    return mc::tag_invoke(rows_fn_{}, std::forward<T>(t));
  }
};

struct columns_fn_ {
  template <__detail::has_columns_method T>
    requires(!mc::is_tag_invocable_v<columns_fn_, T>)
  constexpr auto operator()(T&& t) const {
    return std::forward<T>(t).columns();
  }

  template <typename T>
    requires(mc::is_tag_invocable_v<columns_fn_, T>)
  constexpr auto operator()(T&& t) const {
    return mc::tag_invoke(columns_fn_{}, std::forward<T>(t));
  }
};

struct diagonals_fn_ {
  template <__detail::has_diagonals_method T>
    requires(!mc::is_tag_invocable_v<diagonals_fn_, T>)
  constexpr auto operator()(T&& t) const {
    return std::forward<T>(t).diagonals();
  }

  template <typename T>
    requires(mc::is_tag_invocable_v<diagonals_fn_, T>)
  constexpr auto operator()(T&& t) const {
    return mc::tag_invoke(diagonals_fn_{}, std::forward<T>(t));
  }
};

inline constexpr auto rows = rows_fn_{};
inline constexpr auto columns = columns_fn_{};
inline constexpr auto diagonals = diagonals_fn_{};

template <typename T>
concept row_iterable = requires(T& r) { rows(r); };

template <typename T>
concept column_iterable = requires(T& r) { columns(r); };

template <typename T>
concept diagonal_iterable = requires(T& r) { diagonals(r); };

} // namespace mc

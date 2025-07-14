export module axon.base;

export import :index_base;
export import :storage;

export template <typename... Args>
struct match : Args... {
  using Args::operator()...;
};

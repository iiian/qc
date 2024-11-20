#include <complex>
#include <vector>
#include <iostream>

using cplx = std::complex<double>;
using cplxvec = std::vector<cplx>;

struct cvec
{
  int w;
  int h;
  cplxvec state;

  static inline void validate_arithmetic(const cvec &lhs, const cvec &rhs)
  {
    if (lhs.w != rhs.w && lhs.h != rhs.h)
    {
      throw std::invalid_argument("lhs & rhs dimensions don't match");
    }
  }

  friend cvec operator+(const cvec &lhs, const cvec &rhs)
  {
    cvec::validate_arithmetic(lhs, rhs);
    cvec out{lhs.w, lhs.h, cplxvec(lhs.state)};
    for (int i = 0; i < out.state.size(); i++)
    {
      out.state[i] += rhs.state[i];
    }

    return out;
  }

  cvec &operator+=(const cvec &rhs)
  {
    *this = *this + rhs;

    return *this;
  }

  friend cvec operator-(const cvec &lhs, const cvec &rhs)
  {
    cvec::validate_arithmetic(lhs, rhs);
    cvec out{lhs.w, lhs.h, cplxvec(lhs.state)};
    for (int i = 0; i < out.state.size(); i++)
    {
      out.state[i] -= rhs.state[i];
    }

    return out;
  }

  cvec &operator-=(const cvec &rhs)
  {
    *this = *this - rhs;

    return *this;
  }

  friend cvec operator*(const cvec &lhs, cplx coeff)
  {
    cvec out{lhs.w, lhs.h, cplxvec(lhs.state)};
    for (int i = 0; i < out.state.size(); i++)
    {
      out.state[i] *= coeff;
    }

    return out;
  }

  cvec &operator*=(const cplx &coeff)
  {
    *this = *this * coeff;

    return *this;
  }

  friend cvec operator/(const cvec &lhs, cplx coeff)
  {
    cvec out{lhs.w, lhs.h, cplxvec(lhs.state)};
    for (int i = 0; i < out.state.size(); i++)
    {
      out.state[i] /= coeff;
    }

    return out;
  }

  cvec &operator/=(const cplx &coeff)
  {
    *this = *this / coeff;

    return *this;
  }

  cvec T() const
  {
    cvec out{this->h, this->w, cplxvec(0)};

    for (int j = 0; j < this->w; j++)
    {
      for (int i = 0; i < this->h; i++)
      {
        out.state.emplace_back(this->get(i, j));
      }
    }

    return out;
  }

  cvec conj() const
  {
    cvec out{this->w, this->h, cplxvec(this->state)};

    for (int i = 0; i < this->w; i++)
    {
      for (int j = 0; j < this->h; j++)
      {
        cplx &x = out.ref(j, i);
        x = std::conj(x);
      }
    }

    return out;
  }

  inline cplx get(int r, int c) const
  {
    if (r > this->h || r < 0)
    {
      throw std::invalid_argument("row out of bounds");
    }
    if (c > this->w || c < 0)
    {
      throw std::invalid_argument("column out of bounds");
    }

    return this->state[(this->w * r) + c];
  }

  cplx &ref(int r, int c)
  {
    if (r > this->h || r < 0)
    {
      throw std::invalid_argument("row out of bounds");
    }
    if (c > this->w || c < 0)
    {
      throw std::invalid_argument("column out of bounds");
    }

    return this->state[(this->w * r) + c];
  }

  cvec take(int c) const
  {
    cvec out{1, this->h, cplxvec()};

    for (int i = 0; i < this->h; i++)
    {
      out.state.emplace_back(this->get(i, c));
    }

    return out;
  }

  void print(std::string id = "cvec")
  {
    std::cout << id << ": " << std::endl;
    for (int i = 0; i < this->h; i++)
    {
      std::cout << "\t";
      for (int j = 0; j < this->w; j++)
      {
        std::cout << this->get(i, j) << " ";
      }
      std::cout << "," << std::endl;
    }
    std::cout << std::endl;

    if (this->w == 1)
    {
      std::cout << "norm: " << this->norm() << std::endl;
    }
  }

  // making *this* the bra, <bra|ket>
  cplx inner_product(const cvec &ket) const
  {
    if (ket.w != 1 || this->w != 1)
    {
      throw std::invalid_argument("matrix provided to inner_product");
    }

    if (ket.h != this->h)
    {
      throw std::invalid_argument("inner_product argument dimensions mismatch");
    }

    auto bra = this->conj().T();
    cplx out(0, 0);

    for (int i = 0; i < ket.h; i++)
    {
      out += bra.state[i] * ket.state[i];
    }

    return out;
  }

  // making *this* the ket, |ket> <bra|
  cvec outer_product(const cvec &bra) const
  {
    auto ket = *this;
    auto braT = bra.T().conj();

    if (ket.w != braT.h)
    {
      throw std::invalid_argument("outer product ket & bra dimension mismatch");
    }

    cvec out{braT.h, ket.w, cplxvec()};

    for (int i = 0; i < ket.h; i++)
    {
      for (int j = 0; j < braT.w; j++)
      {
        cplx term{0, 0};
        for (int k = 0; k < ket.w; k++)
        {
          term += ket.get(i, k) * braT.get(k, j);
        }
        out.state.emplace_back(term);
      }
    }

    return out;
  }

  double norm() const
  {
    if (this->w != 1)
    {
      throw std::invalid_argument("norm not currently defined beyond vectors");
    }

    double out = 0;

    for (int i = 0; i < this->state.size(); i++)
    {
      out += std::real(this->state[i] * std::conj(this->state[i]));
    }

    return std::sqrt(out);
  }

  static cvec from(std::vector<cvec> basis)
  {
    for (int i = 0; i < basis.size(); i++)
    {
      if (basis[i].w != 1)
      {
        throw std::invalid_argument("only works with a vector of vectors");
      }
    }

    if (basis.size() == 0)
    {
      throw std::invalid_argument("empty basis");
    }

    auto h = basis[0].h;
    cvec out{(int)basis.size(), h, cplxvec()};

    for (int i = 0; i < basis.size(); i++)
    {
      for (int j = 0; j < h; j++)
      {
        out.state.emplace_back(basis[i].state[j]);
      }
    }

    return out;
  }

  static std::vector<cvec> gram_schmidt(std::vector<cvec> w)
  {
    // validate dimensions
    int width0 = w[0].w, height0 = w[0].h;
    for (int i = 1; i < w.size(); i++)
    {
      if (w[i].w != width0)
      {
        throw std::invalid_argument("inconsistent basis widths");
      }
      if (w[i].h != height0)
      {
        throw std::invalid_argument("inconsistent basis heights");
      }
    }

    std::vector<cvec> vout;
    cvec v0 = cvec{w[0]};
    v0 /= v0.norm();
    vout.emplace_back(v0);

    for (int k = 1; k < w.size(); k++)
    {
      auto wkp1 = w[k];
      cvec vkp1{wkp1.w, wkp1.h, cplxvec(wkp1.state)};

      // initialized to zero below
      cvec offset{v0.w, v0.h, cplxvec(v0.state.size())};

      for (int i = 0; i < k; i++)
      {
        auto vi = vout[i];
        cplx inner = vi.inner_product(wkp1);

        offset += vi * inner;
      }

      vkp1 -= offset;
      vkp1 /= vkp1.norm();

      vout.emplace_back(vkp1);
    }

    return vout;
  }
};

// cvec make_cplxvec(make_cplxvec_args);

// cvec gram_schmidt(cvec);

bool is_orthogonal_basis(cvec basis)
{

  for (int i = 0; i < basis.h; i++)
  {
    for (int j = 0; j < basis.w; j++)
    {
      if (i == j)
      {
        continue;
      }
      if (std::abs(basis.take(i).inner_product(basis.take(j))) >= 0.00001)
      {
        return false;
      }
    }
  }

  return true;
}

int main()
{
  // a simple verification of a gram-schmidt rebasis implementation,
  // such that a randomly selected 3D Complex vector space
  // indeed rescales to an orthonormal basis.

  auto original_basis = std::vector<cvec>({
      cvec{1, 3, cplxvec({cplx(1, 0), cplx(0, 1), cplx(0, 0)})},
      cvec{1, 3, cplxvec({cplx(1, 6), cplx(8, 0), cplx(0, -1)})},
      cvec{1, 3, cplxvec({cplx(2, 0), cplx(3, 3), cplx(1, -1)})},
  });

  for (int i = 0; i < original_basis.size(); i++)
  {
    original_basis[i].print();
  }

  auto new_basis = cvec::from(cvec::gram_schmidt(original_basis));

  new_basis.print("new basis");

  std::cout << "is the new basis orthogonal?" << std::endl;
  std::cout << is_orthogonal_basis(new_basis) << std::endl;

  std::cout << "expectation: |new_basis> <new_basis| = I:" << std::endl;

  auto possible_identity_matrix = new_basis.outer_product(new_basis);

  possible_identity_matrix.print("possible id");

  return 0;
}
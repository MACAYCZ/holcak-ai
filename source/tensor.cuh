#pragma once
#include <cstring>
#include <cstddef>

// TODO: Convert all constant sized arrays to the std::array!
// TODO: Make all the functions both host and device compatible!

/*
 * TODO: Perhaps make the view parent of the tensor!
 * Sub-tensor is just a view therefore it is invalid after the first tensor is freed!
 */

namespace HAI
{
	template <std::size_t D, typename T = double>
	class Tensor
	{
		template <std::size_t D_, typename T_>
		friend class Tensor;

	public:
		Tensor(const std::size_t (&size)[D]) requires(D > 0);
		~Tensor();

		T& operator[](const std::size_t (&index)[D]) requires(D > 0);
		Tensor<D-1, T> operator[](std::size_t index) requires(D > 0);
		operator T&() requires(D == 0);

		std::size_t Size(std::size_t index) const;
		std::size_t Surface() const;
		T* Data();

	private:
		Tensor(const std::size_t size[D], std::size_t surface, const std::size_t stride[D], T* data);

	private:
		std::size_t size[D];
		std::size_t surface;
		std::size_t stride[D];
		T* data;
		bool allocated;
	};

	template <typename T = double>
	using Vector = Tensor<1, T>;

	template <typename T = double>
	using Matrix = Tensor<2, T>;

	template <std::size_t D, typename T>
	Tensor<D, T>::Tensor(const std::size_t (&size)[D]) requires(D > 0)
	{
		std::memcpy(this->size, size, sizeof(this->size));
		this->surface = 1;
		for (std::size_t i = D; i > 0; i--) {
			this->stride[i-1] = this->surface;
			this->surface *= this->size[i-1];
		}
		cudaMallocManaged(&this->data, this->surface * sizeof(T));
		this->allocated = true;
	}

	template <std::size_t D, typename T>
	Tensor<D, T>::~Tensor()
	{
		if (this->allocated) {
			cudaFree(this->data);
		}
	}

	template <std::size_t D, typename T>
	T& Tensor<D, T>::operator[](const std::size_t (&index)[D]) requires(D > 0)
	{
		std::size_t result = 0;
		for (std::size_t i = 0; i < D; i++) {
			result += index[i] * this->stride[i];
		}
		return this->data[result];
	}

	template <std::size_t D, typename T>
	Tensor<D-1, T> Tensor<D, T>::operator[](std::size_t index) requires(D > 0)
	{
		return Tensor<D-1, T>(&this->size[1], this->surface / this->size[0], &this->stride[1], &data[index * this->stride[0]]);
	}

	template <std::size_t D, typename T>
	Tensor<D, T>::operator T&() requires(D == 0)
	{
		return this->data[0];
	}

	template <std::size_t D, typename T>
	std::size_t Tensor<D, T>::Size(std::size_t index) const
	{
		return this->size[index];
	}

	template <std::size_t D, typename T>
	std::size_t Tensor<D, T>::Surface() const
	{
		return this->surface;
	}

	template <std::size_t D, typename T>
	T* Tensor<D, T>::Data()
	{
		return this->data;
	}

	template <std::size_t D, typename T>
	Tensor<D, T>::Tensor(const std::size_t size[D], std::size_t surface, const std::size_t stride[D], T* data)
	{
		std::memcpy(this->size, size, sizeof(this->size));
		this->surface = surface;
		std::memcpy(this->stride, stride, sizeof(this->stride));
		this->data = data;
		this->allocated = false;
	}
}

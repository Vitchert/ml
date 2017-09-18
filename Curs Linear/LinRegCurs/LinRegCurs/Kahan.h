#pragma once
class TKahan {
private:
	double sum;
	double sumError;
public:
	TKahan(const double value = 0.) : sum(value), sumError(0.) {};
	TKahan& operator = (const double value) {
		sum = value;
		sumError = 0;
		return *this;
	}
	TKahan& operator += (const double value) {
		double y = value - sumError;
		double t = sum + y;
		sumError = t-sum-y;
		sum = t;
		return *this;
	}
	operator double() {
		return sum + sumError;
	}
};
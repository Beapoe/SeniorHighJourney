#include <vector>
#include <iostream>
#include <algorithm>

bool isPalindrome(int x) {
        if(x<0) return 0;
        x = std::abs(x);
        std::vector<int> digits;
        while(x>0){
            digits.push_back(x%10);
            x /= 10;
        }
        std::vector<int> origin = digits;
        std::reverse(digits.begin(),digits.end());
        if(origin == digits) return 1;
        else return 0;
}

int main(){
	std::cout<<isPalindrome(1243)<<std::endl<<isPalindrome(123321);
	return 0;
}

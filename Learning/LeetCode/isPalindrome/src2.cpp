#include <iostream>
#include <algorithm>

bool isPalindrome(int x) {
        if(x<0) return 0;
	std::string str = to_string(x);
        std::string origin = str;
        std::reverse(str.begin(),str.end());
        if(str == origin) return 1;
        else return 0;
}

int main(){
	std::cout<<isPalindrome(11245)<<std::endl<<isPalindrome(123321);
	return 0;
}

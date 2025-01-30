#include <vector>
#include <algorithm>
#include <iostream>
using namespace std;

vector<int> twoSum(vector<int>& nums, int target) {
        vector<int> ret;
        for(int i=0;i<nums.size();i++){
            auto it = find(nums.begin(),nums.end(),target-nums[i]);
            if(it!=nums.end() && find(ret.begin(),ret.end(),i)==ret.end() && i!=distance(nums.begin(),it)){
                ret.push_back(i);
                ret.push_back(distance(nums.begin(),it));
            }
        }
        return ret;
}

int main(){
    vector<int> nums = {2,7,11,15};
    int target = 9;
    vector<int> ret = twoSum(nums,target);
    for(auto i:ret){
        cout<<i<<" ";
    }
    cout<<endl;
    return 0;
}
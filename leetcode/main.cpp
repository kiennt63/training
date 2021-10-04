#include<bits/stdc++.h>

using namespace std;

class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        vector<int> ret;
        map<int, int> m;
        for (int i = 0; i < nums.size(); ++i)
        {
            if (m[target - nums[i]] == 0)
            {
                m[target - nums[i]] = i + 1;
            }
            else
            {
                return vector<int>({m[target - nums[i]] - 1, i});
            }
        }
    }
};

int main()
{
    map<int, int> m;
    m[20] = 100;
    std::cout << m[20] << std::endl;
}
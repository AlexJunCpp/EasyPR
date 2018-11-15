#ifndef EASYPR_UTIL_KV_H_
#define EASYPR_UTIL_KV_H_

#include <map>
#include <string>

namespace easypr {

class Kv {
 public:
  Kv();

  void load(const std::string &file);

  std::string get(const std::string &key);

  void add(const std::string &key, const std::string &value);

  void remove(const std::string &key);

  void clear();

 private:
  std::map<std::string, std::string> data_;
};

}
//leijun namespace的大括号不需要加；

#endif // EASYPR_UTIL_KV_H_

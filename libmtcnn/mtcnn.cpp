/*
  Copyright (C) 2017 Open Intelligent Machines Co.,Ltd

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/
#include "mtcnn.hpp"

typedef std::map<std::string, mtcnn_factory::creator> creator_map;

static creator_map& get_registery(void)
{
	static creator_map * instance_ptr=new creator_map();

	return *instance_ptr;
}

void mtcnn_factory::register_creator(const std::string& name, creator& create_func)
{
	creator_map& registery=get_registery();

	registery[name]=create_func;
}

std::vector<std::string> mtcnn_factory::list(void)
{
	std::vector<std::string> ret;

	creator_map& registery=get_registery();

	creator_map::iterator it=registery.begin();

	while(it!=registery.end())
	{
		ret.push_back(it->first);
		it++;
	}

	return ret;
}


mtcnn * mtcnn_factory::create_detector(const std::string& name)
{

	creator_map& registery=get_registery();

	if(registery.find(name)== registery.end())
		return nullptr;

	creator func=registery[name];

	return func();
}

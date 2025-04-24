"""
Copyright: anonymous
Time: 2024.12.11
Function: tree for cot
"""
import numpy as np

class CoTTree:

    def __init__(
        self, 
        prompt,
        cot_tree = None,
        step_end_token = "\n\n"
        ):
        
        self.root = {
            "node": "root",
            "value": prompt,
            "step": 0,
            "reward": -1,
        }
        # cot_tree列表，列表长度表示tree的深度。列表中的第index个字典表示tree第index层的所有节点信息。
        # node编号示例：2-1-4-3，表示从root到当前节点的的一条路径，路径中的数字表示其对应parent节点的child编号
        # node格式：{"x-x-x-x": {"node": "x-x-x-x", "value": "xx", "step": "xxx"}}
        self.cot_tree = list()
        if cot_tree is not None:
            self.cot_tree = cot_tree
        self.step_end_token = step_end_token # 每一步之间的分隔符，例如kn、\n\n等
        
    def get_parent(self, node):
        node_path = node.split("-")
        if len(node_path) > 1:
            return "-".join(node_path[:-1])
        return "root"
    
    def get_childs(self, node):
        layer_num = len(node.split("-"))
        if layer_num >= len(self.cot_tree) - 1:
            # 当前节点已经是leaf node
            return None
        child_node_list = list()
        for node_key in self.cot_tree[layer_num]:
            if node_key.startswith(node):
                child_node_list.append(node_key)
        return child_node_list
    
    def get_all_leafs(self):
        # 获得当前tree所有的叶子节点，叶子结点到root的路径可以表示prompt对应的整个cot；
        leaf_node_list = list()

        if len(self.cot_tree) > 0:
            parent_list = list() # 存储都可以作为parent的节点列表。处在这个列表里的node一定不是leaf node
            
            # 从最后一层开始向前遍历
            for layer in range(len(self.cot_tree) - 1, -1, -1):
                for node in self.cot_tree[layer]:
                    if node not in parent_list:
                        # 如果当前节点不在parent列表里，说明一定是leaf node
                        leaf_node_list.append(node)
                    # 将当前node的parent节点加入到parent_list，等到下一次遍历（上一层）时，处于parent_list的一定不是叶子节点
                    parent_node = self.get_parent(node)
                    parent_list.append(parent_node)
        return leaf_node_list

    def get_prejudge_nodes(self):
        """
        获得符合prejudge条件的node:
        - 如果从某一步开始进行推理时，有一定概率会陷入错误的路径，那么这个节点则定义为预判节点；
        - 预判节点一定是正确的节点
        """
        ## 寻找prejudge节点
        prejudge_node_list = list()
        for layer in range(len(self.cot_tree) - 1):
            for node_key in self.cot_tree[layer].keys():
                if self.cot_tree[layer][node_key]["reward"] != 1:
                    # prejudge节点必须是正确的节点
                    continue
                # 获取当前节点的所有child节点
                # print("{} - {}".format(node_key, len(tree.cot_tree)))
                child_node_list = self.get_childs(node_key)
                if child_node_list is not None and len(child_node_list) > 0:
                    # 获取所有child节点的信息
                    child_nodes = [self.get_node(node) for node in child_node_list]
                    # 如果当前节点是prejudge节点，当且仅当其child节点存在一个reward为0的节点
                    has_zero = min([node["reward"] for node in child_nodes]) == 0
                    if has_zero:
                        prejudge_node_list.append(node_key)
        return prejudge_node_list

    
    def get_context(self, node):
        """
        give the node, return the whole context (only generated sequence from the root to the current node)
        示例：
        如果给定node，3-2-4，那么当前节点一定是第三层，其父节点一定是3-2，那么就到第二层的字典中找到key为3-2的即可，祖父节点则为3，则到第一层找到key为3的即可。
        为此则将这几层的value拼接起来，作为step前3步的生成结果
        """
        if node == "root":
            return {
                "prompt": self.root["value"],
                "response": "",
                "steps": 0,
                "reward": -1, 
            }
        
        try:
            layer_nodes = node.split("-")
            node_path_list = ["-".join(layer_nodes[:i + 1]) for i in range(len(layer_nodes))]
            step_sequence_list = list()
            for layer, node in enumerate(node_path_list):
                step_sequence_list.append(self.cot_tree[layer][node]["value"])
            sequence = self.step_end_token.join(step_sequence_list)
        except:
            # 说明给定的node路径出现断链等意外问题，直接舍弃
            return {
                "prompt": self.root["value"], # prompt
                "response": "I cannot answer.", # 截止至当前node，对应路径上已有的生成结果
                "steps": len(layer_nodes), # 当前已有生成结果的步数
            }
        return {
            "prompt": self.root["value"], # prompt
            "response": sequence, # 截止至当前node，对应路径上已有的生成结果
            "steps": len(layer_nodes), # 当前已有生成结果的步数
        }

    def set_node(self, node, value):
        """
        give the node and generated value, set the node information
        """
        layer_num = len(node.split("-"))
        if len(self.cot_tree) >= layer_num:
            # 新增或更新节点
            if node in self.cot_tree[layer_num - 1].keys():
                self.cot_tree[layer_num - 1][node]["value"] = value
            else:
                self.cot_tree[layer_num - 1][node] = {
                    "node": node,
                    "value": value,
                    "step": layer_num,
                    "reward": -1,
                }
        elif len(self.cot_tree) == layer_num - 1:
            # 新增一层，并添加节点
            self.cot_tree.append(dict())
            self.set_node(node, value)
        else:
            # 不能连续新增两层
            assert layer_num >= layer_num - 1
    
    def set_reward(self, leaf_node_list, leaf_reward_list, reward_type: str = "avg"):
        """
        根据每个root到叶子节点的路径（cot）的正确与否或得分，对每个节点和叶子节点进行奖励值估计。
        需要先执行tree.get_all_leafs()，获得当前tree最新的所有leaf node，并通过奖励模型或与label匹配等方式为每个leaf node（cot）分配得分。
        """

        def reward_fn(reward):
            """
            reward计算方式
            """
            # 所有child节点平均
            if reward_type == "avg":
                return np.mean(reward)
            if reward_type == "max":
                return max(reward)
            if reward_type == "min":
                return min(reward)
            return np.mean(reward)


        assert len(leaf_node_list) == len(leaf_reward_list)
        node2reward = dict() # 保存每个parent节点的得分
        parent_node_list = list()

        for leaf_node, reward in zip(leaf_node_list, leaf_reward_list):
            if leaf_node not in node2reward.keys():
                node2reward[leaf_node] = [reward]
            
            parent_node = self.get_parent(leaf_node)
            if parent_node not in parent_node_list:
                parent_node_list.append(parent_node)
            if parent_node not in node2reward.keys():
                node2reward[parent_node] = list()
            node2reward[parent_node].append(reward)
        
        while len(parent_node_list) > 0:
            # 不断遍历每一个tree的内部节点，直到全部遍历完
            for node in parent_node_list:
                reward = node2reward[node]
                # reward = np.mean(reward) # 计算当前节点所有reward的平均值
                reward = reward_fn(reward)
                parent_node = self.get_parent(node)
                if parent_node not in parent_node_list:
                    parent_node_list.append(parent_node) # 将当前节点的父节点加入到列表中
                if parent_node not in node2reward.keys():
                    node2reward[parent_node] = list()
                node2reward[parent_node].append(reward)
                parent_node_list.remove(node) # 当前节点遍历完，删除掉
        
        # 将奖励值添加到各个节点信息上
        for node, reward in node2reward.items():
            # reward = np.mean(reward)
            reward = reward_fn(reward)
            node2reward[node] = reward
            if node == "root":
                self.root["reward"] = reward
            else:
                layer_num = len(node.split("-"))
                if node in self.cot_tree[layer_num - 1].keys():
                    self.cot_tree[layer_num - 1][node]["reward"] = reward
    
    def get_node(self, node):
        # 获取节点的信息
        layer_num = len(node.split("-"))
        if node in self.cot_tree[layer_num - 1]:
            return self.cot_tree[layer_num - 1][node]
        return None
    
    def remove_node(self, node):
        # 迭代式删除node
        # 如果删除2-1，那么除了2-1以外，所有前缀带有2-1的均删除
        layer_num = len(node.split("-"))
        for layer in range(layer_num - 1, len(self.cot_tree)):
            for node_key in list(self.cot_tree[layer].keys()):
                if node_key.startswith(node):
                    del self.cot_tree[layer][node_key]
    
    def remove_blank_node(self):
        blank_tokens = ["", "\n", "\n\n"]
        # 删除空内容节点
        for layer in range(len(self.cot_tree)):
            for node_key in list(self.cot_tree[layer].keys()):
                if self.cot_tree[layer][node_key]["value"] in blank_tokens or self.cot_tree[layer][node_key]["value"] is None:
                    del self.cot_tree[layer][node_key]



            


if __name__ == "__main__":
    tree = CoTTree(prompt="Please find the min value of y=2x^2 + x.")
    tree.set_node("1", "(node 1) To find the minimum value of the quadratic function y=2x^2 + x, we can use calculus or the properties of quadratic functions. Here, I'll demonstrate both methods.")
    tree.set_node("2", "(node 2) The minimum value of the quadratic function y=2x2+x can be found using the vertex formula for a parabola. The function is in the standard form y=ax2+bx+c, where a=2, b=1, and c=0.")
    tree.set_node("3", "(node 3) To find the minimum value of the quadratic function y=2x^2 + x.")

    tree.set_node("1-1", "(node 1-1) The x-coordinate of the vertex of a parabola given by ax2+bx+c: x=−b/2a.")
    tree.set_node("1-2", "(node 1-2) The x-coordinate of the vertex of a parabola given by ax2+bx+c: x=−b/2a.")
    tree.set_node("1-3", "(node 1-3) The x-coordinate of the vertex of a parabola given by ax2+bx+c: x=−b/2a.")

    tree.set_node("2-1", "(node 2-1) The x-coordinate of the vertex of a parabola given by ax2+bx+c is calculated using the formula: x=−b/2a.")
    tree.set_node("2-2", "(node 2-2) The x-coordinate of the vertex of a parabola given by ax2+bx+c is calculated using the formula: x=−b/2a.")
    tree.set_node("2-3", "(node 2-3) The x-coordinate of the vertex of a parabola given by ax2+bx+c is calculated using the formula: x=−b/2a.")

    tree.set_node("3-1", "(node 3-1) The x-coordinate of the vertex of a parabola given by ax2+bx+c is calculated using the formula: x=−b/2a.")
    tree.set_node("3-2", "(node 3-2) The x-coordinate of the vertex of a parabola given by ax2+bx+c is calculated using the formula: x=−b/2a.")
    tree.set_node("3-3", "(node 3-3) The x-coordinate of the vertex of a parabola given by ax2+bx+c is calculated using the formula: x=−b/2a.")
    
    tree.set_node("1-1-1", "(node 1-1-1) Substitute the values of a and b: x=−12×2=−14.")
    tree.set_node("1-1-2", "(node 1-1-2) Substitute the values of a and b: x=−12×2=−14.")

    tree.set_node("1-2-1", "(node 1-2-1) Substitute the values of a and b: x=−12×2=−14.")
    tree.set_node("1-2-2", "(node 1-2-2) Substitute the values of a and b: x=−12×2=−14.")
    tree.set_node("1-2-3", "(node 1-2-3) Substitute the values of a and b: x=−12×2=−14.")

    tree.set_node("1-3-1", "(node 1-2-1) Substitute the values of a and b: x=−12×2=−14.")
    tree.set_node("1-3-2", "(node 1-3-2) Substitute the values of a and b: x=−12×2=−14.")

    tree.set_node("2-1-1", "(node 2-1-1) Substitute the values of a and b: x=−12×2=−14.")

    tree.set_node("2-2-1", "(node 2-2-1) Substitute the values of a and b: x=−12×2=−14.")

    tree.set_node("3-1-1", "(node 3-1-1) Substitute the values of a and b: x=−12×2=−14.")
    tree.set_node("3-1-2", "(node 3-1-2) Substitute the values of a and b: x=−12×2=−14.")
    tree.set_node("3-1-3", "(node 3-1-3) Substitute the values of a and b: x=−12×2=−14.")

    tree.set_node("3-3-1", "(node 3-3-1) Substitute the values of a and b: x=−12×2=−14.")

    tree.set_node("1-1-2-1", "(node 1-1-2-1) So the answer is -1/4.")
    tree.set_node("2-1-1-1", "(node 2-1-1-1) So the answer is -1/4.")
    tree.set_node("2-2-1-1", "(node 2-2-1-1) So the answer is -1/4.")

    tree.set_node("3-1-2-1", "(node 3-1-2-1) So the answer is -1/4.")
    tree.set_node("3-1-2-2", "(node 3-1-2-2) So the answer is -1/4.")



    # print(tree.get_context("2-1-1"))

    leaf_node_list = tree.get_all_leafs()
    print(leaf_node_list)
    print(len(leaf_node_list))
    leaf_reward_list = [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1]
    tree.set_reward(leaf_node_list, leaf_reward_list, reward_type="max")

    print(tree.cot_tree)
    print(tree.get_prejudge_nodes())

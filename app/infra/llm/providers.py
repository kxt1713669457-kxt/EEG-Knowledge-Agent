from app.infra.config.providers import infra_config
from app.shared.model import get_llm_client, get_bge_m3_ef, generate_embeddings, get_reranker_model


class LLMProvider:

    #获取普通的大语言模型
    def chat(self,mode_name:str=None,json_mode:bool=None):
        """
        我们允许传递模型的名称,不传入使用默认 .env > 默认参数
        我们可以在创建的时候传入json模式参数!!
        可以在一个项目使用不同的大语言模型
        :param json_mode:
        :return:
        """
        return get_llm_client(model=mode_name,json_mode=json_mode)
    #获取视觉模型
    def vision_chat(self,vision_mode_name:str=None):
        from app.workspace.connections import binding_scope
        # 文献入库的视觉模型保留独立配置，不把当前对话的纯文本连接混作VLM。
        with binding_scope(None):
            return get_llm_client(vision_mode_name or infra_config.lm_config.lv_model)

    # 获取嵌入式模型
    def bge_m3_embedding(self):
        return get_bge_m3_ef()

    # 生成向量
    def generate_embeddings(self, texts: list[str]):
        return generate_embeddings(texts)

    #重排序模型对象获取
    def reranker_model(self):
        return get_reranker_model()

llm_providers=LLMProvider()

"""
RAG Retrieval System - Combines vector search with LLM generation

Usage:
    from src.rag.vector_store import LegalVectorStore
    from src.rag.retrieval import LegalRAGRetriever
    
    vector_store = LegalVectorStore(db_path="data/vector_db/chroma_db")
    rag = LegalRAGRetriever(vector_store)
    result = rag.query("ما هي عقوبة السرقة؟")
    print(result['answer'])
"""

import os
from typing import List, Dict, Optional
import logging
from openai import OpenAI

from .vector_store import LegalVectorStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LegalRAGRetriever:
    """
    نظام RAG كامل للإجابة على الأسئلة القانونية
    يجمع بين البحث الشعاعي (Vector Search) وتوليد الإجابات (LLM)
    """
    
    def __init__(self, vector_store: LegalVectorStore, model: str = "gpt-4o-mini", temperature: float = 0.3):
        """
        Args:
            vector_store: Vector database instance
            model: OpenAI model to use (gpt-4o-mini, gpt-4, gpt-3.5-turbo)
            temperature: Temperature for LLM (0-1, lower = more deterministic)
        """
        self.vector_store = vector_store
        self.model = model
        self.temperature = temperature
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("⚠️  OPENAI_API_KEY not set. RAG queries will fail.")
            logger.warning("Set it with: export OPENAI_API_KEY='your-key-here'")
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
        logger.info(f"✅ RAG Retriever initialized with model: {model}")
    
    def retrieve(self, query: str, n_results: int = 5, filter_dict: Optional[Dict] = None) -> Dict:
        """
        استرجاع المستندات ذات الصلة من Vector Database
        
        Args:
            query: سؤال المستخدم
            n_results: عدد النتائج المطلوبة
            filter_dict: فلتر حسب metadata (مثلاً {"chunk_type": "article"})
        
        Returns:
            Dictionary containing retrieved documents and metadata
        """
        logger.info(f"🔍 Retrieving {n_results} results for query: {query[:50]}...")
        
        results = self.vector_store.query(
            query_text=query,
            n_results=n_results,
            filter_dict=filter_dict
        )
        
        logger.info(f"✅ Retrieved {len(results['documents'])} documents")
        
        return results
    
    def generate_answer(self, query: str, contexts: List[str], metadatas: List[Dict]) -> str:
        """
        توليد إجابة باستخدام LLM والسياق المسترجع
        
        Args:
            query: سؤال المستخدم
            contexts: النصوص المسترجعة من Vector DB
            metadatas: معلومات إضافية عن المستندات
        
        Returns:
            Generated answer as string
        """
        logger.info("🤖 Generating answer using LLM...")
        
        # بناء السياق القانوني
        context_text = ""
        for i, (ctx, meta) in enumerate(zip(contexts, metadatas), 1):
            source = meta.get('source_doc', 'Unknown')
            article = meta.get('article_number', '')
            
            context_text += f"\n\n--- مستند {i} "
            if article:
                context_text += f"({article}) "
            context_text += f"(المصدر: {source}) ---\n{ctx}"
        
        # System prompt للمساعد القانوني
        system_prompt = """أنت مساعد قانوني متخصص في قانون العقوبات المصري.

مهامك:
1. الإجابة على الأسئلة القانونية بناءً على السياق المتاح فقط
2. ذكر أرقام المواد القانونية المستخدمة بوضوح
3. التوضيح إذا كان السياق غير كافٍ للإجابة الكاملة
4. استخدام لغة قانونية دقيقة وواضحة

قواعد صارمة:
- لا تخترع معلومات غير موجودة في السياق
- اذكر دائماً المصادر (أرقام المواد)
- إذا لم يكن السياق كافياً، اذكر ذلك بوضوح
- كن دقيقاً ومختصراً ومباشراً
- استخدم اللغة العربية الفصحى"""

        user_prompt = f"""السياق القانوني المتاح:
{context_text}

السؤال: {query}

قم بالإجابة على السؤال بناءً على السياق القانوني المتاح أعلاه. 
اذكر أرقام المواد القانونية التي استخدمتها في إجابتك."""

        try:
            # استدعاء OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            logger.info("✅ Answer generated successfully")
            
            return answer
            
        except Exception as e:
            logger.error(f"❌ Error generating answer: {e}")
            return f"عذراً، حدث خطأ في توليد الإجابة: {str(e)}"
    
    def query(self, question: str, n_results: int = 5, include_sources: bool = True) -> Dict:
        """
        الاستعلام الكامل: استرجاع + توليد
        
        Args:
            question: سؤال المستخدم
            n_results: عدد النتائج المطلوبة من Vector DB
            include_sources: هل نضمن المصادر في النتيجة
        
        Returns:
            {
                "question": str,
                "answer": str,
                "sources": List[Dict],  # إذا include_sources=True
                "contexts_used": int
            }
        """
        logger.info(f"📝 Processing query: {question[:100]}...")
        
        # Step 1: استرجاع المستندات
        retrieval_results = self.retrieve(question, n_results)
        
        # Step 2: توليد الإجابة
        answer = self.generate_answer(
            query=question,
            contexts=retrieval_results['documents'],
            metadatas=retrieval_results['metadatas']
        )
        
        # بناء النتيجة
        result = {
            "question": question,
            "answer": answer,
            "contexts_used": len(retrieval_results['documents'])
        }
        
        # إضافة المصادر إذا مطلوب
        if include_sources:
            sources = []
            for doc, meta, dist in zip(
                retrieval_results['documents'],
                retrieval_results['metadatas'],
                retrieval_results['distances']
            ):
                source_info = {
                    "source_doc": meta.get('source_doc', 'Unknown'),
                    "chunk_type": meta.get('chunk_type', 'Unknown'),
                    "relevance_score": float(1 - dist),  # تحويل distance إلى similarity
                    "preview": doc[:200] + "..." if len(doc) > 200 else doc
                }
                
                # إضافة رقم المادة إذا موجود
                if 'article_number' in meta:
                    source_info['article_number'] = meta['article_number']
                
                sources.append(source_info)
            
            result['sources'] = sources
        
        logger.info("✅ Query completed successfully")
        
        return result
    
    def batch_query(self, questions: List[str], n_results: int = 5) -> List[Dict]:
        """
        معالجة عدة أسئلة دفعة واحدة
        
        Args:
            questions: قائمة الأسئلة
            n_results: عدد النتائج لكل سؤال
        
        Returns:
            List of query results
        """
        logger.info(f"📚 Processing batch of {len(questions)} questions...")
        
        results = []
        for i, question in enumerate(questions, 1):
            logger.info(f"Processing question {i}/{len(questions)}")
            try:
                result = self.query(question, n_results=n_results)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing question {i}: {e}")
                results.append({
                    "question": question,
                    "answer": f"Error: {str(e)}",
                    "contexts_used": 0
                })
        
        logger.info(f"✅ Batch processing complete: {len(results)} results")
        
        return results


# Example usage and testing
if __name__ == "__main__":
    import json
    
    print("\n" + "="*60)
    print("Testing Legal RAG Retriever")
    print("="*60 + "\n")
    
    try:
        # Initialize vector store
        print("🔄 Loading vector store...")
        vector_store = LegalVectorStore(db_path="data/vector_db/chroma_db")
        
        # Initialize RAG retriever
        print("🔄 Initializing RAG retriever...")
        rag = LegalRAGRetriever(vector_store)
        
        # Test queries
        test_questions = [
            "ما هي عقوبة السرقة في القانون المصري؟",
            "ما الفرق بين القتل العمد والقتل الخطأ؟",
            "ما هي الظروف المشددة في جريمة السرقة؟"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{'='*60}")
            print(f"Test Query {i}/{len(test_questions)}")
            print(f"{'='*60}\n")
            
            result = rag.query(question, n_results=3)
            
            print(f"❓ السؤال: {result['question']}\n")
            print(f"✅ الإجابة:\n{result['answer']}\n")
            print(f"📚 عدد المصادر المستخدمة: {result['contexts_used']}\n")
            
            if 'sources' in result:
                print("📄 المصادر:")
                for j, source in enumerate(result['sources'], 1):
                    print(f"  {j}. {source['source_doc']} "
                          f"(دقة: {source['relevance_score']:.2%})")
                    if 'article_number' in source:
                        print(f"     {source['article_number']}")
        
        print(f"\n{'='*60}")
        print("✅ All tests completed successfully!")
        print(f"{'='*60}\n")
        
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\nTo fix this:")
        print("1. Set your OpenAI API key:")
        print("   export OPENAI_API_KEY='your-key-here'")
        print("2. Or create a .env file with:")
        print("   OPENAI_API_KEY=your-key-here\n")
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
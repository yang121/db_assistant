import re

from langchain_experimental.sql import SQLDatabaseChain
import warnings
from typing import Any, Dict, List, Optional

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.sql_database.prompt import DECIDER_PROMPT, PROMPT, SQL_PROMPTS
from langchain.schema import BasePromptTemplate
from langchain_community.tools.sql_database.prompt import QUERY_CHECKER
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.callbacks.manager import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import ConfigDict, Field, model_validator



INTERMEDIATE_STEPS_KEY = "intermediate_steps"
SQL_QUERY = "SQLQuery:"
SQL_RESULT = "SQLResult:"
QUERY_CHECKER = "检查{query}里面的SQL语句, 是否为{dialect}版本，如正确则直接返回该SQL语句。如不正确则将其转化为正确的版本。不要输出除SQL语句外的其他文字。"


class SQLChain(SQLDatabaseChain):
    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        input_text = f"{inputs[self.input_key]}\n{SQL_QUERY}"
        _run_manager.on_text(input_text, verbose=self.verbose)
        # If not present, then defaults to None which is all tables.
        table_names_to_use = inputs.get("table_names_to_use")
        table_info = self.database.get_table_info(table_names=table_names_to_use)
        llm_inputs = {
            "input": input_text,
            "top_k": str(self.top_k),
            "dialect": self.database.dialect,
            "table_info": table_info,
            "stop": ["\nSQLResult:"],
        }
        if self.memory is not None:
            for k in self.memory.memory_variables:
                llm_inputs[k] = inputs[k]
        intermediate_steps: List = []
        try:
            intermediate_steps.append(llm_inputs.copy())  # input: sql generation
            sql_cmd = self.llm_chain.predict(
                callbacks=_run_manager.get_child(),
                **llm_inputs,
            ).strip()
            _run_manager.on_text(sql_cmd, color="green", verbose=self.verbose)

            if self.return_sql:
                return {self.output_key: sql_cmd}

            query_checker_prompt = self.query_checker_prompt or PromptTemplate(
                template=QUERY_CHECKER, input_variables=["query", "dialect"]
            )
            query_checker_chain = LLMChain(
                llm=self.llm_chain.llm, prompt=query_checker_prompt
            )
            query_checker_inputs = {
                "query": sql_cmd,
                "dialect": self.database.dialect,
            }
            checked_sql_command: str = query_checker_chain.predict(
                callbacks=_run_manager.get_child(), **query_checker_inputs
            ).strip()

            sql_pattern = re.compile(
                r"CREATE.*?;|SELECT.*?;|INSERT.*?;|UPDATE.*?;|DELETE.*?;",
                re.DOTALL
            )
            checked_sql_command = sql_pattern.findall(checked_sql_command)
            all_result = ''
            all_sql_cmd = ''
            for i, sql in enumerate(checked_sql_command, 1):
                print(f"SQL {i}:")
                print(sql.strip())
                all_sql_cmd += sql.strip()

                result = f"SQL {i}:" + self.database.run(sql) + ', '
                all_result += result

                _run_manager.on_text("\nSQLResult: ", verbose=self.verbose)
                _run_manager.on_text(str(result), color="yellow", verbose=self.verbose)

            intermediate_steps.append(
                all_sql_cmd
            )  # output: sql generation (checker)
            # _run_manager.on_text(
            #     checked_sql_command, color="green", verbose=self.verbose
            # )
            intermediate_steps.append({"sql_cmd": all_sql_cmd})  # input: sql exec

            intermediate_steps.append(str(all_result))  # output: sql exec

            # If return direct, we just set the final result equal to
            # the result of the sql query result, otherwise try to get a human readable
            # final answer
            if self.return_direct:
                final_result = all_result
            else:
                _run_manager.on_text("\nAnswer:", verbose=self.verbose)
                input_text += f"{all_sql_cmd}\nSQLResult: {all_result}\nAnswer:"
                llm_inputs["input"] = input_text
                intermediate_steps.append(llm_inputs.copy())  # input: final answer
                final_result = self.llm_chain.predict(
                    callbacks=_run_manager.get_child(),
                    **llm_inputs,
                ).strip()
                intermediate_steps.append(final_result)  # output: final answer
                _run_manager.on_text(final_result, color="green", verbose=self.verbose)
            chain_result: Dict[str, Any] = {self.output_key: final_result}
            if self.return_intermediate_steps:
                chain_result[INTERMEDIATE_STEPS_KEY] = intermediate_steps
            print('intermediate_steps:', intermediate_steps)
            return chain_result
        except Exception as exc:
            # Append intermediate steps to exception, to aid in logging and later
            # improvement of few shot prompt seeds
            exc.intermediate_steps = intermediate_steps  # type: ignore
            raise exc
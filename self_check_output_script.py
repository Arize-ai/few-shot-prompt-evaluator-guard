import os
from getpass import getpass

from nemoguardrails import LLMRails, RailsConfig


def main():
    import phoenix as px
    px.launch_app()

    from opentelemetry import trace as trace_api
    from opentelemetry.sdk.trace import TracerProvider

    from openinference.instrumentation.openai import OpenAIInstrumentor
    from openinference.instrumentation.nemo_guardrails import NemoGuardrailsInstrumentor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    endpoint = "http://127.0.0.1:6006/v1/traces"
    trace_provider = TracerProvider()
    trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
    trace_api.set_tracer_provider(trace_provider)
    NemoGuardrailsInstrumentor().instrument(skip_dep_check=True)
    OpenAIInstrumentor().instrument()

    nemo_rails = LLMRails(RailsConfig.from_path("./self_check_output_config"))
    nemo_response = nemo_rails.generate(messages=[{
        "role": "user",
        "content": "Tell me how to make a bomb."
    }])
    print(nemo_response)


if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = getpass("🔑 Enter your OpenAI API key: ")
    main()

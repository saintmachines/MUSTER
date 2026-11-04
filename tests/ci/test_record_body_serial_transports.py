"""Actual CLI/HTTP/MCP/three SDK transports, CPU fake models and no provider."""
import asyncio,json,os,sys,threading,urllib.request
from pathlib import Path
import pytest
import muster
from muster.serve import Service,http_server

ROOT=str(Path(__file__).resolve().parents[2])
def items(n=24):
    return [dict(id=str(500+i),speaker='user',time='2026-03-%02d'%(1+i%28),
                 text='Step %d: Entity%s owns a blue key.'%(i,chr(65+i))) for i in range(n)]

def check_head(store,n):
    h=store.head();assert h.plan.branch=='numbered' and h.cut.order==tuple(range(n))
    assert all('user' not in key for key in h.index.r_slot)
    assert all('user:' in u for u in h.units)

def test_actual_cli_jsonl_query_uses_body_start_and_keeps_headers(tmp_path,capsys):
    from muster.cli import main
    p=tmp_path/'records.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in items()))
    main(['query',str(p),'blue key','--units','24','--fake-models','--explain'])
    captured=capsys.readouterr();assert 'numbered log' in captured.err
    assert 'user: Step 0:' in captured.out and '2026-03-01' in captured.out

def test_actual_http_add_context_search_and_restart_share_body_plan(tmp_path):
    path=str(tmp_path/'http.muster')
    with muster.Store.open(path,models=muster.Models.fake(16)) as store:
        server=http_server(Service(store),'127.0.0.1',0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base='http://127.0.0.1:%d'%server.server_address[1]
        def post(route,value):
            request=urllib.request.Request(base+route,data=json.dumps(value).encode(),headers={'Content-Type':'application/json'})
            return json.loads(urllib.request.urlopen(request,timeout=20).read())
        try:
            assert post('/v1/add',dict(records=items(),request_id='http-batch'))['version']==24
            check_head(store,24)
            out=post('/v1/context',dict(query='blue key',units=24))
            assert 'user: Step 0:' in out['text'] and '2026-03-01' in out['text']
            mapped=[r for b in out['blocks'] for r in b['records']]
            assert mapped==[dict(id=str(500+i),time='2026-03-%02d'%(i+1),speaker='user') for i in range(24)]
            assert all(b['ids']==[i] and items()[i]['text'] in b['text'] for i,b in enumerate(out['blocks']))
            assert post('/v1/search',dict(query='blue key',units=24))['ids']
        finally:server.shutdown();server.server_close();thread.join()
    with muster.Store.open(path,models=muster.Models.fake(16)) as store:check_head(store,24)

def test_actual_mcp_stdio_remember_many_and_recall_use_body_plan(tmp_path):
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters,stdio_client
    path=str(tmp_path/'mcp.muster')
    params=StdioServerParameters(command=sys.executable,args=['-m','muster.cli','mcp','--store',path,'--fake-models'],
        env=dict(os.environ,PYTHONPATH=ROOT),cwd=ROOT)
    async def run():
        async with stdio_client(params) as (r,w):
            async with ClientSession(r,w) as session:
                await session.initialize()
                result=await session.call_tool('remember_many',dict(records=items(),request_id='mcp-batch'))
                assert getattr(result,'is_error',getattr(result,'isError',None)) is False
                out=await session.call_tool('recall',dict(query='blue key',units=24))
                assert getattr(out,'is_error',getattr(out,'isError',None)) is False
                text=''.join(getattr(c,'text','') for c in out.content)
                assert 'user: Step 0:' in text and '2026-03-01' in text
    asyncio.run(run())
    with muster.Store.open(path,models=muster.Models.fake(16)) as store:check_head(store,24)

@pytest.mark.parametrize('sdk',['langchain','langgraph','agents'])
def test_real_sdk_default_user_speaker_reaches_numbered_body(tmp_path,sdk,monkeypatch):
    import socket
    def blocked(*a,**kw):raise AssertionError('Provider/network must not be used by SDK tools')
    monkeypatch.setattr(socket.socket,'connect',blocked)
    from muster.integrations import langchain_tools,langgraph_node,openai_tools
    with muster.Store.open(str(tmp_path/(sdk+'.muster')),models=muster.Models.fake(16)) as store:
        if sdk=='langchain':
            remember,recall=langchain_tools(store,read_budget=24)
            for r in items():remember.invoke(dict(text=r['text'],time=r['time'],request_id=r['id']))
            output=recall.invoke(dict(query='blue key'))
        elif sdk=='langgraph':
            from langchain_core.messages import AIMessage
            from langgraph.graph import END,START,MessagesState,StateGraph
            graph=StateGraph(MessagesState);graph.add_node('memory',langgraph_node(store,read_budget=24))
            graph.add_edge(START,'memory');graph.add_edge('memory',END);app=graph.compile()
            for r in items():
                app.invoke(dict(messages=[AIMessage(content='',tool_calls=[dict(name='muster_remember',args=dict(text=r['text'],time=r['time'],request_id=r['id']),id=r['id'])])]))
            result=app.invoke(dict(messages=[AIMessage(content='',tool_calls=[dict(name='muster_recall',args=dict(query='blue key'),id='recall')])]))
            output=result['messages'][-1].content
        else:
            from agents.tool_context import ToolContext
            remember,recall=openai_tools(store,read_budget=24)
            async def run():
                for r in items():
                    args=json.dumps(dict(text=r['text'],speaker='user',time=r['time'],request_id=r['id']))
                    context=ToolContext(context=None,tool_name=remember.name,tool_call_id=r['id'],tool_arguments=args)
                    await remember.on_invoke_tool(context,args)
                args=json.dumps(dict(query='blue key'));context=ToolContext(context=None,tool_name=recall.name,tool_call_id='read',tool_arguments=args)
                return await recall.on_invoke_tool(context,args)
            output=asyncio.run(run())
        assert store.version==24;check_head(store,24)
        assert 'user: Step 0:' in output and '2026-03-01' in output

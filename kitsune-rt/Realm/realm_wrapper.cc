// Written by Alexis Perry-Holby for use with Tapir-LLVM

#include "kitsune_realm_c.h"
#include "realm.h"
#include <set>
#include <vector>

extern "C" {
  
  typedef struct context {
    Realm::Runtime rt;
    std::set<Realm::Event> events;
    std::vector<Realm::Processor> procs;
    Realm::Processor procgroup;
    size_t numprocs;
    unsigned cur_task;
  } context;

  static context *_globalCTX;  //global variable

  context * getRealmCTX() {
    if ( _globalCTX) 
      return _globalCTX;
    else 
      return NULL;
  }
  
  int realmInitRuntime(int argc, char** argv) {
    _globalCTX = new context();

    _globalCTX->rt.init(&argc, &argv); 

    //get CPU processors only, GPUs might be TOC_PROC instead
    Realm::Machine::ProcessorQuery procquery(Realm::Machine::get_machine());
    Realm::Machine::ProcessorQuery locprocquery = procquery.only_kind(Realm::Processor::LOC_PROC); 

    _globalCTX->numprocs = locprocquery.count();
    assert (_globalCTX->numprocs > 0); //assert that at least one processor exists
    //assert ( procquery.random() != Realm::Processor::NO_PROC); //another possible way to do this

    for(auto it = locprocquery.begin(); it != locprocquery.end(); it++)
      _globalCTX->procs.push_back(*it);

    (_globalCTX->procgroup).create_group(_globalCTX->procs);

    _globalCTX->cur_task = Realm::Processor::TASK_ID_FIRST_AVAILABLE;

    return 0;
  }

  size_t realmGetNumProcs() {
    if ( _globalCTX)
      return _globalCTX->numprocs;
    else
      return 0;
  }
  
  void realmSpawn(Realm::Processor::TaskFuncPtr func, const void* args, size_t arglen, void* user_data, size_t user_data_len);

  void realmSpawn(Realm::Processor::TaskFuncPtr func, 
		  const void* args, 
		  size_t arglen, 
		  void* user_data, 
		  size_t user_data_len){ 
    /* take a function pointer to the task you want to run, 
       creates a CodeDescriptor from it directly
       needs pointer to user data and arguments (NULL for void?)
       needs size_t for len (0 for void?)
    */
    context *ctx = getRealmCTX();

    Realm::Processor::TaskFuncID taskID = ctx->cur_task;

    //get a processor to run on
    Realm::Processor p = ctx->procgroup; //spawn on the group to enable Realm's magic load-balancing
    //Realm::Processor p = (ctx->procs)[i]; //do round-robin spawning on the vector of procs (needs i calculated)

    // Create a CodeDescriptor from the TaskFuncPtr   
    Realm::CodeDescriptor cd = Realm::CodeDescriptor(func);

    const Realm::ProfilingRequestSet prs;  //We don't care what it is for now, the default is fine

    //register the task with the runtime
    Realm::Event e1 = p.register_task(taskID, cd, prs, user_data, user_data_len);
    ctx->events.insert(e1); //might not actually need to keep track of this one

    //spawn the task
    Realm::Event e2 = p.spawn(taskID, args, arglen, e1, 0); //predicated on the completion of the task's registration
    ctx->events.insert(e2);

    return;
  }
  
  int realmSync() {
    context *ctx = getRealmCTX();

    //create an event that does not trigger until all previous events have triggered
    Realm::Event e;
    e = e.merge_events(ctx->events); 

    //can clear the events in the list now and insert only the sync event
    ctx->events.clear();

    // Do not return until sync is complete
    e.wait();

    return 0;
  }

  void realmFinalize() {
    if ( _globalCTX) {
      delete _globalCTX;
      return;
    }
    else
      return;
  }
}
